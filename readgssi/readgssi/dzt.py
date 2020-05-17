import struct
import math
import os
import numpy as np
from datetime import datetime
from itertools import takewhile
from readgssi.gps import readdzg
import readgssi.functions as fx
from readgssi.constants import *
from readgssi.dzx import get_user_marks, get_picks

"""
helper module for reading information from GSSI DZT files

the main function, readdzt(), returns header information in dictionary format
and the radar profile in a numpy array.

readdzt_gprpy() restructures output for use specifically with
Alain Plattner's GPRPy software (https://github.com/NSGeophysics/GPRPy).
"""

def readtime(bytes):
    """
    Function to read dates from the :code:`rfDateByte` binary objects in DZT headers. 

    DZT :code:`rfDateByte` objects are 32 bits of binary (01001010111110011010011100101111), structured as little endian u5u6u5u5u4u7 where all numbers are base 2 unsigned int (uX) composed of X number of bits. It's an unnecessarily high level of compression for a single date object in a filetype that often contains tens or hundreds of megabytes of array information anyway.

    So this function reads (seconds/2, min, hr, day, month, year-1980) then does seconds*2 and year+1980 and returns a datetime object.

    For more information on :code:`rfDateByte`, see page 55 of `GSSI's SIR 3000 manual <https://support.geophysical.com/gssiSupport/Products/Documents/Control%20Unit%20Manuals/GSSI%20-%20SIR-3000%20Operation%20Manual.pdf>`_.

    :param bytes bytes: The :code:`rfDateByte` to be decoded
    :rtype: :py:class:`datetime.datetime`
    """
    dtbits = ''
    rfDateByte = (b for b in bytes)
    for byte in rfDateByte:                    # assemble the binary string
        for i in range(8):
            dtbits += str((byte >> i) & 1)
    dtbits = dtbits[::-1]               # flip the string
    sec2 = int(dtbits[27:32], 2) * 2    # seconds are stored as seconds/2 because there's only 5 bytes to work with
    mins = int(dtbits[21:27], 2)        # minutes
    hr = int(dtbits[16:21], 2)          # hours
    day = int(dtbits[11:16], 2)         # day
    mo = int(dtbits[7:11], 2)           # month
    yr = int(dtbits[0:7], 2) + 1980     # year, stored as 1980+(0:127)
    return datetime(yr, mo, day, hr, mins, sec2, 0, tzinfo=pytz.UTC)

def readdzt(infile, gps=False, spm=None, start_scan=0, num_scans=-1, epsr=None, antfreq=[None,None,None,None], verbose=False):
    """
    Function to unpack and return things the program needs from the file header, and the data itself.

    :param str infile: The DZT file location
    :param bool gps: Whether a GPS file exists. Defaults to False, but changed to :py:class:`pandas.DataFrame` if a DZG file with the same name as :code:`infile` exists.
    :param float spm: User value of samples per meter, if specified. Defaults to None.
    :param float epsr: User value of relative permittivity, if specified. Defaults to None.
    :param bool verbose: Verbose, defaults to False
    :rtype: header (:py:class:`dict`), radar array (:py:class:`numpy.ndarray`), gps (False or :py:class:`pandas.DataFrame`)
    """

    '''
    currently unused but potentially useful lines:
    # headerstruct = '<5h 5f h 4s 4s 7h 3I d I 3c x 3h d 2x 2c s s 14s s s 12s h 816s 76s' # the structure of the bytewise header and "gps data" as I understand it - 1024 bytes
    # readsize = (2,2,2,2,2,4,4,4,4,4,2,4,4,4,2,2,2,2,2,4,4,4,8,4,3,1,2,2,2,8,1,1,14,1,1,12,2) # the variable size of bytes in the header (most of the time) - 128 bytes
    # fx.printmsg('total header structure size: '+str(calcsize(headerstruct)))
    # packed_size = 0
    # for i in range(len(readsize)): packed_size = packed_size+readsize[i]
    # fx.printmsg('fixed header size: '+str(packed_size)+'\\n')
    '''
    infile_gps = os.path.splitext(infile)[0] + ".DZG"
    infile_dzx = os.path.splitext(infile)[0] + ".DZX"
    infile = open(infile, 'rb')
    header = {}
    header['infile'] = infile.name
    header['known_ant'] = [None, None, None, None]
    header['rh_ant'] = [None, None, None, None]
    header['rh_antname'] = [None, None, None, None]
    header['antfreq'] = [None, None, None, None]

    # begin read

    header['rh_tag'] = struct.unpack('<h', infile.read(2))[0] # 0x00ff if header, 0xfnff if old file format
    header['rh_data'] = struct.unpack('<h', infile.read(2))[0] # offset to data from beginning of file
    header['rh_nsamp'] = struct.unpack('<h', infile.read(2))[0] # samples per scan
    header['rh_bits'] = struct.unpack('<h', infile.read(2))[0] # bits per data word
    header['rh_zero'] = struct.unpack('<h', infile.read(2))[0] # if sir-30 or utilityscan df, then repeats per sample; otherwise 0x80 for 8bit and 0x8000 for 16bit
    header['rhf_sps'] = struct.unpack('<f', infile.read(4))[0] # scans per second
    header['rhf_spm'] = struct.unpack('<f', infile.read(4))[0] # scans per meter
    header['dzt_spm'] = header['rhf_spm']
    if spm:
        header['rhf_spm'] = spm

    header['rhf_mpm'] = struct.unpack('<f', infile.read(4))[0] # meters per mark
    header['rhf_position'] = struct.unpack('<f', infile.read(4))[0] # position (ns)
    header['rhf_range'] = struct.unpack('<f', infile.read(4))[0] # range (ns)
    header['rh_npass'] = struct.unpack('<h', infile.read(2))[0] # number of passes for 2-D files
    # bytes 32-36 and 36-40: creation and modification date and time in bits
    # structured as little endian u5u6u5u5u4u7
    infile.seek(32)
    try:
        header['rhb_cdt'] = readtime(infile.read(4))
    except:
        header['rhb_cdt'] = datetime(1980, 1, 1)
    try:
        header['rhb_mdt'] = readtime(infile.read(4))
    except:
        header['rhb_mdt'] = datetime(1980, 1, 1)
    header['rh_rgain'] = struct.unpack('<h', infile.read(2))[0] # offset to range gain function
    header['rh_nrgain'] = struct.unpack('<h', infile.read(2))[0] # size of range gain function
    header['rh_text'] = struct.unpack('<h', infile.read(2))[0] # offset to text
    header['rh_ntext'] = struct.unpack('<h', infile.read(2))[0] # size of text
    header['rh_proc'] = struct.unpack('<h', infile.read(2))[0] # offset to processing history
    header['rh_nproc'] = struct.unpack('<h', infile.read(2))[0] # size of processing history
    header['rh_nchan'] = struct.unpack('<h', infile.read(2))[0] # number of channels
    if epsr != None: # in this case the user has specified an epsr value
        header['dzt_epsr'] = struct.unpack('<f', infile.read(4))[0]
        header['rhf_epsr'] = epsr
    else:
        header['rhf_epsr'] = struct.unpack('<f', infile.read(4))[0] # epsr (sometimes referred to as "dielectric permittivity")
        header['dzt_epsr'] = header['rhf_epsr']
    header['rhf_top'] = struct.unpack('<f', infile.read(4))[0] # position in meters (useless?)
    header['dzt_depth'] = struct.unpack('<f', infile.read(4))[0] # range in meters based on DZT rhf_epsr
    header['rhf_depth'] = header['dzt_depth'] * (math.sqrt(header['dzt_epsr']) / math.sqrt(header['rhf_epsr'])) # range based on user epsr
    #rhf_coordx = struct.unpack('<ff', infile.read(8))[0] # this is definitely useless

    freq = [None, None, None, None]
    for i in range(header['rh_nchan']):
        try:
            freq[i] = antfreq[i]
        except (TypeError, IndexError):
            freq[i] = 200

    # read frequencies for multiple antennae
    for chan in list(range(header['rh_nchan'])):
        if chan == 0:
            infile.seek(98) # start of antenna section
        else:
            infile.seek(98 + (MINHEADSIZE*(chan))) # start of antenna bytes for channel n
        header['rh_ant'][chan] = infile.read(14).decode('utf-8').split('\x00')[0]
        header['rh_antname'][chan] = header['rh_ant'][chan].rsplit('x')[0]
        try:
            header['antfreq'][chan] = ANT[header['rh_antname'][chan]]
            header['known_ant'][chan] = True
        except KeyError:
            header['known_ant'][chan] = False
            try:
                header['antfreq'][chan] = int("".join(takewhile(str.isdigit, header['rh_ant'][chan].replace('D5','').replace('D6','')))) # hoping this works
            except ValueError:
                header['antfreq'] = freq
            #header['antfreq'][chan] = int(header['rh_antname'][chan].replace('D5','').replace('D6',''))

    infile.seek(113) # skip to something that matters
    vsbyte = infile.read(1) # byte containing versioning bits
    header['rh_version'] = ord(vsbyte) >> 5 # whether or not the system is GPS-capable, 1=no 2=yes (does not mean GPS is in file)
    header['rh_system'] = ord(vsbyte) >> 3 # the system type (values in UNIT={...} dictionary above)

    if header['rh_system'] == 14:   # hardcoded because this is so frustrating. assuming no other antennas can be paired with SS Mini XT
        header['rh_antname'] = ['SSMINIXT', None, None, None]
        header['antfreq'] = [2700, None, None, None]
        header['known_ant'] = [True, False, False, False]

    infile.seek(header['rh_rgain'])
    try:
        header['rgain_bytes'] = infile.read(header['rh_nrgain'])
    except:
        pass

    if header['rh_data'] < MINHEADSIZE: # whether or not the header is normal or big-->determines offset to data array
        infile.seek(MINHEADSIZE * header['rh_data'])
        header['data_offset'] = MINHEADSIZE * header['rh_data']
    else:
        infile.seek(MINHEADSIZE * header['rh_nchan'])
        header['data_offset'] = MINHEADSIZE * header['rh_nchan']

    if header['rh_bits'] == 8:
        dtype = np.uint8 # 8-bit unsigned
    elif header['rh_bits'] == 16:
        dtype = np.uint16 # 16-bit unsigned
    else:
        dtype = np.int32 # 32-bit signed
    
    if start_scan != 0:
        try:
            # calculate start offset in bytes:
            start_offset = int(start_scan * header['rh_nchan'] * header['rh_nsamp'] * header['rh_bits']/8)
        except ValueError:
            # if this fails, then fall back to 0 offset.
            start_offset = 0
            fx.printmsg('WARNING: ValueError for scan offset: {start_scan} (reading from start of data)')
            # consider returning No Data?
    else:
        start_offset = 0
    
    if num_scans != -1:
        try:
            num_items = int(num_scans * header['rh_nsamp']*header['rh_nchan'])
        except ValueError:
            # if this fails then get all scans...
            fx.printmsg('WARNING: ValueError for number of scans: {num_scans} (reading all items from {start_scan} scans)')
            num_items = -1
    else:
        num_items = -1
            
    # read in and transpose data
    data = np.fromfile(infile, dtype, count=num_items).reshape(-1,(header['rh_nsamp']*header['rh_nchan'])).T # offset=start_offset,

    header['cr'] = 1 / math.sqrt(Mu_0 * Eps_0 * header['rhf_epsr'])
    header['cr_true'] = 1 / math.sqrt(Mu_0 * Eps_0 * header['dzt_epsr'])
    header['ns_per_zsample'] = (header['rhf_depth'] * 2) / (header['rh_nsamp'] * header['cr'])
    header['samp_freq'] = 1 / ((header['dzt_depth'] * 2) / (header['rh_nsamp'] * header['cr_true']))

    try:
        header['sec'] = data.shape[1]/float(header['rhf_sps'])
    except ZeroDivisionError:
        header['sec'] = 1.

    infile.close()

    if os.path.isfile(infile_gps):
        try:
            if verbose:
                fx.printmsg('reading GPS file...')
            gps = readdzg(infile_gps, 'dzg', header, verbose=verbose)
        except IOError as e0:
            fx.printmsg('WARNING: cannot read DZG file')
            try:
                infile_gps = os.path.splitext(infile_gps)[0] + ".csv"
                gps = readdzg(infile_gps, 'csv', header, verbose=verbose)
            except Exception as e1:
                try:
                    infile_gps = os.path.splitext(infile_gps)[0] + ".CSV"
                    gps = readdzg(infile_gps, 'csv', header, verbose=verbose)
                except Exception as e2:
                    fx.printmsg('ERROR reading GPS. distance normalization will not be possible.')
                    fx.printmsg('   details: %s' % e0)
                    fx.printmsg('            %s' % e1)
                    fx.printmsg('            %s' % e2)
                    gps = []
    else:
        fx.printmsg('WARNING: no DZG file found for GPS input')

    header['marks'] = []
    header['picks'] = {}

    if os.path.isfile(infile_dzx):
        header['marks'] = get_user_marks(infile_dzx, verbose=verbose)
        header['picks'] = get_picks(infile_dzx, verbose=verbose)
    else:
        fx.printmsg('WARNING: could not find DZX file to read metadata. Trying to read array for marks...')

        tnums = np.ndarray.tolist(data[0])  # the first row of the array is trace number
        usr_marks = np.ndarray.tolist(data[1])  # when the system type is SIR3000, the second row should be user marks (otherwise these are in the DZX, see note below)
        i = 0
        for m in usr_marks:
            if m > 0:
                print(m)
                header['marks'].append(i)
            i += 1
        fx.printmsg('DZT marks read successfully. marks: %s' % len(header['marks']))
        fx.printmsg('                            traces: %s' % header['marks'])

    return [header, data, gps]

def readdzt_gprpy(infile):
    r = readdzt(infile)
    data = r[1]
    header = {
        'sptrace': r[0]['rh_nsamp'],
        'scpsec': r[0]['rhf_sps'],
        'scpmeter': r[0]['rhf_spm'],
        'startposition': r[0]['rhf_position'],
        'nanosecptrace': r[0]['rhf_range'],
        'scansppass': r[0]['rh_npass'],
    }
    return data, header

def header_info(header, data):
    """
    Function to print relevant header data.

    :param dict header: The header dictionary
    :param numpy.ndarray data: The data array
    """
    fx.printmsg('system:             %s (system code %s)' % (UNIT[header['rh_system']], header['rh_system']))
    fx.printmsg('antennas:           %s' % header['rh_antname'])
    fx.printmsg('ant freqs. (MHz):   %s' % header['antfreq'])
    fx.printmsg('ant time zeros:     %s' % header['timezero'])

    for i in range(header['rh_nchan']):
        if header['known_ant'][i] == True:
            fx.printmsg('ant %s center freq:  %s MHz' % (i, ANT[header['rh_antname'][i]]))
        else:
            fx.printmsg('ant %s center freq:  %s MHz (antenna name %s not in current dictionary)'
                        % (i, header['antfreq'][i], header['rh_antname'][i]))

    fx.printmsg('date created:       %s' % header['rhb_cdt'])
    if header['rhb_mdt'] == datetime(1980, 1, 1):
        fx.printmsg('date modified:      (never modified)')
    else:
        fx.printmsg('date modified:      %s' % header['rhb_mdt'])
    try:
        fx.printmsg('gps-enabled file:   %s' % GPS[header['rh_version']])
    except (TypeError, KeyError) as e:
        fx.printmsg('gps-enabled file:   unknown')
    fx.printmsg('number of channels: %i' % header['rh_nchan'])
    fx.printmsg('samples per trace:  %i' % header['rh_nsamp'])
    fx.printmsg('bits per sample:    %s' % BPS[header['rh_bits']])
    fx.printmsg('traces per second:  %.1f' % header['rhf_sps'])
    if header['dzt_spm'] != header['rhf_spm']:
        fx.printmsg('traces per meter:   %.2f (manually set - value from DZT: %.2f)' % (float(header['rhf_spm']), float(header['dzt_spm'])))
    else:
        fx.printmsg('traces per meter:   %.1f' % (float(header['rhf_spm'])))
    if header['dzt_epsr'] != header['rhf_epsr']:
        fx.printmsg('user epsr:          %.1f (manually set - value from DZT: %.1f)' % (header['rhf_epsr'], header['dzt_epsr']))
    else:
        fx.printmsg('epsr:               %.1f' % header['rhf_epsr'])
    fx.printmsg('speed of light:     %.2E m/sec (%.2f%% of vacuum)' % (header['cr'], header['cr'] / C * 100))
    if header['dzt_depth'] != header['rhf_depth']:
        fx.printmsg('sampling depth:     %.1f m (manually set - value from DZT: %.1f)' % (header['rhf_depth'], header['dzt_depth']))
    else:
        fx.printmsg('sampling depth:     %.1f m' % (header['rhf_depth']))
    fx.printmsg('"rhf_top":          %.1f m' % header['rhf_top'])
    fx.printmsg('offset to data:     %i bytes' % header['data_offset'])
    if data.shape[1] == int(data.shape[1]):
        fx.printmsg('traces:             %i' % int(data.shape[1]/header['rh_nchan']))
    else:
        fx.printmsg('traces:             %f' % int(data.shape[1]/header['rh_nchan']))
    fx.printmsg('seconds:            %.8f' % (header['sec']))
    fx.printmsg('array dimensions:   %i x %i' % (data.shape[0], data.shape[1]))
