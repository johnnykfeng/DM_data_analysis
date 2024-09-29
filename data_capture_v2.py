import os
import time
import socket
import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy import io as sio

import easygui as eg
from easygui import easygui

import pcct
import settings
from fat.calibrations import electrical_cal_filename, energy_cal_filename, write_csv_electrical_cal, \
    write_csv_energy_cal
from fat.data import CCData, bin2array
from fat.instruments import Instruments
from pcct.buffer import Buffer, BufferTimeoutError
from pcct.ddc0864.constants import MAX_THRESHOLD, N_THRESHOLDS
from pcct.fpga.dmb import DmbFpga
from pcct_canon import CanonDm8Xmed, CanonDm8XmedRevE


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# ###############################################################################################################
# This script depends on dm-tp libraries and should be copied to main dm-tp folder(feature/508-monitoring-avdd1 branch).
# The raw and processed data will be saved in the same directory where the script lies.
# ###############################################################################################################
#TEST_TYPE = 'RISING_EDGE'  # or 'THRESHOLD_SWEEP'  # TODO: change test type
#TEST_TYPE = 'RISING_EDGE'
TEST_TYPE = 'XRAY_SWEEP'

if TEST_TYPE == 'XRAY_SWEEP':
    FRAMES = 4
    VIEW_PERIOD = 1000
    RESOLUTION = 0.250

elif TEST_TYPE == 'RISING_EDGE':
    FRAMES = 400
    VIEW_PERIOD = 1000
    RESOLUTION = 0.0025
    SHUTTER_ON = 0.5
    SHUTTER_OFF = 0.5
    TOTAL_TIME = 2
    capture_delay_time = 0.4                      # [s]

BIAS = 1000
THRESHOLDS = [20, 30, 50, 70, 90, 120]  # TODO: change bin thresholds for xray sweep


# TODO: change respective to calib file directory at C:\CTData\{TANK_SERIAL}\docs\deviceSpecificCals\{asic_serial}
TANK_SERIAL = r'S1160 Red Box'
CALIB_DIR = f'C:\CTData\EC4352\docs\deviceSpecificCals'

#   TODO: change DM serial and enter MM numbers for desired slots
SERIAL = 'S1160'
MODULES = {
    0: 'M58618',
    1: 'M58747',
    2: 'M62543',
    3: 'M62546',
    4: 'M63667',
    5: 'M63668',
    6: 'M63670',
    7: 'M63671',
}

FOLDER_NAME = {'XRAY_SWEEP': 'XrayCapture', 'THRESHOLD_SWEEP': 'ThresholdScan'}
COUNTERS = {'EC': True, 'CC': True, 'SEC': False}
##############################################################################################################

WORKING_DIR = os.getcwd()
# baseline = settings.DDC0864_BASELINES[settings.DDC0864_BASELINES]
baseline = settings.baseline('B')
ENERGY_CAL = True


asic_serials = {}
for module_idx, module_serial in MODULES.items():
    asic_serials[module_idx * 2] = f'{module_serial}-A0'
    asic_serials[module_idx * 2 + 1] = f'{module_serial}-A1'

asics_in_use = list(asic_serials.keys())


def prepare_output_data(fpga, num_sweep_dim=1, threshold_params=None, tube_currents=None, tube_voltages=None, filters=None):
    """Generate output data structures for each sensor."""
    counters = fpga.n_counters
    frames = FRAMES
    asic_data_structs = {}

    for asic_idx in fpga.selected_asics:
        cc_data = CCData(num_sweep_dim, counters, frames)
        cc_data.params['test_type'] = TEST_TYPE
        cc_data.params['baseline'] = baseline['name']
        cc_data.params['view_period'] = VIEW_PERIOD
        cc_data.params['resolution'] = RESOLUTION
        cc_data.params['frames'] = FRAMES
        cc_data.params['thresholds'] = THRESHOLDS if threshold_params is None else []
        cc_data.params['counters'] = COUNTERS
        cc_data.params['hv_bias'] = BIAS
        cc_data.params['threshold_start'] = threshold_params[0] if threshold_params is not None else []
        cc_data.params['threshold_end'] = threshold_params[1] if threshold_params is not None else []
        cc_data.params['threshold_step'] = threshold_params[2] if threshold_params is not None else []


        cc_data.params['tube_current'] = tube_currents if tube_currents is not None else []
        cc_data.params['tube_voltage'] = tube_voltages if tube_voltages is not None else []
        cc_data.params['filter'] = filters if filters is not None else []

        asic_data_structs[asic_idx] = cc_data

    return asic_data_structs


def generate_sweep_thresholds(start, step):
    thresholds = []
    threshold = start
    for _ in range(N_THRESHOLDS):
        thresholds.append(threshold if threshold <= MAX_THRESHOLD else MAX_THRESHOLD)
        threshold += step

    return thresholds


def make_pwm_string(on, off, total_time):
    """creates pwm string for shutter control
    calculates number of shutter cycles according to on/off periods
    increments counter while adding off / on time until reaching totTime"""
    shutter_cycles = 0
    time_accum = 0

    # close-open-close as one cycle
    while time_accum < total_time:
        time_accum += off
        if time_accum < total_time:
            time_accum += on
            if time_accum < total_time:
                time_accum += off
                shutter_cycles += 1

                # in case the user inputs a combination of on/off/total_time that requires an additional cycle
                if time_accum <= total_time:
                    shutter_cycles += 1
        else:
            break

    pwm_string = 'pwm {} {} {}'.format(on, off, shutter_cycles)

    return pwm_string


def sweep_thresholds(fpga, asic_data_structs, sweep, step=1, start_time=None):
    # Generate sweep thresholds
    sweep_len = len(sweep)

    for sweep_idx, sweep_threshold in enumerate(sweep):
        thresholds = generate_sweep_thresholds(sweep_threshold, step)
        for _, asic in fpga.iter_asics_by_version():
            asic.write_thresh_chains(thresholds=thresholds)

        logger.info("Thresholds %02d/%02d: %s AU", sweep_idx + 1, sweep_len, ', '.join(str(t) for t in thresholds))
        capture_data(fpga, asic_data_structs, sweep_idx=sweep_idx, start_time=start_time)


def sweep_xray(fpga, asic_data_structs, tube_currents, start_time=None):
    for sweep_idx, tube_current in enumerate(tube_currents):
        eg.msgbox(msg=f'Set tube current to {tube_current} mA.')
        capture_data(fpga, asic_data_structs, sweep_idx=sweep_idx, start_time=start_time)


def rising_edge(fpga, asic_data_structs, tube_currents, instruments, shutter_on=0.5, shutter_off=0.5, total_time=4, start_time=None):
    for sweep_idx, tube_current in enumerate(tube_currents):
        instruments.shutter.put('close')
        eg.msgbox(msg=f'Set tube current to {tube_current} mA.')
        pwm_string = make_pwm_string(shutter_on, shutter_off, total_time)
        instruments.shutter.put(pwm_string)
        time.sleep(capture_delay_time)
        capture_data(fpga, asic_data_structs, sweep_idx=sweep_idx, start_time=start_time)


def capture_data(fpga, asic_data_structs, sweep_idx=0, start_time=None):
    extra_data = {}
    elapsed_time = 0 if start_time is None else time.time() - start_time
    data = fpga.capture(frames=FRAMES + 2, period=VIEW_PERIOD, resolution=RESOLUTION)[2:]
    if settings.ENABLE_MONITORING:
        extra_data.update(fpga.monitoring_data())
    """Store data in output data structures"""
    bin2array(asic_data_structs, sweep_idx, data, FRAMES, RESOLUTION, elapsed_time, extra_data=extra_data)


def plot_and_save_fig(fpga, asic_data_structs, test_type, num_sweep_dim=1, fig_filenames=None, output_path=None,
                      tube_currents=None, thresh_list=None):
    array_map = np.zeros((num_sweep_dim, 6, 24 * 8, 72), dtype='int')
    frames_data = np.zeros((num_sweep_dim, 6, FRAMES, 16), dtype='int')
    if fpga.n_counters == 13:
        # take slice of 6 Coincidence Counters if all counters (EC, CC, SEC) were collected
        counter_range = [6, 12]
    else:
        counter_range = [0, 6]

    for asic_idx in fpga.selected_asics:
        frames_data[:, :, :, asic_idx] = np.median(asic_data_structs[asic_idx].data['cc_data'][:,
                                                   counter_range[0]:counter_range[1], :, :, :], axis=(-1, -2))
        if np.mod(asic_idx, 2) == 0:
            mm_idx = int(asic_idx / 2)
            array_map[:, :, mm_idx * 24:(mm_idx + 1) * 24, :36] = np.rot90(
                np.sum(asic_data_structs[asic_idx].data['cc_data'][:, counter_range[0]:counter_range[1], :, :, :],
                       axis=2), 2, (-1, -2))
        else:
            mm_idx = int((asic_idx - 1) / 2)
            array_map[:, :, mm_idx * 24:(mm_idx + 1) * 24, 36:] = \
                np.sum(asic_data_structs[asic_idx].data['cc_data'][:, counter_range[0]:counter_range[1], :, :, :],
                       axis=2)

    if test_type == 'XRAY_SWEEP':
        plot_xray_sweep_visuals(array_map, frames_data, fpga.selected_asics, tube_currents,
                                fig_filenames=fig_filenames, output_path=output_path)

    elif test_type == 'RISING_EDGE':
        plot_xray_sweep_visuals(array_map, frames_data, fpga.selected_asics, tube_currents,
                                fig_filenames=fig_filenames, output_path=output_path)

    elif test_type == 'THRESHOLD_SWEEP':
        fig_filepath = os.path.join(output_path, fig_filenames)
        plot_threshold_sweep_visuals(frames_data, thresh_list, fig_filepath)


def plot_xray_sweep_visuals(array_map, frames_data, asics, tube_currents, fig_filenames, output_path):
    for current_idx, tube_current in enumerate(tube_currents):
        fig_filepath = os.path.join(output_path, f'{fig_filenames[current_idx]}')
        fig, ax = plt.subplots(1, len(THRESHOLDS) + 1, layout='constrained', figsize=(20.0, 15.0))
        for i in range(len(THRESHOLDS)):
            # calculate and plot raw uncollimated counts
            subplot_name = f'{THRESHOLDS[i]} keV'
            map = array_map[current_idx, i, :, :]
            # plot ratio image
            # plt.subplot(1, len(THRESHOLDS) + 1, i + 1)[0.0,3.0]
            map_range = np.percentile(map, [0.5, 99.5])
            im = ax[i].imshow(map, vmin=map_range[0], vmax=map_range[1])
            plt.colorbar(im, ax=ax[i], label='counts')
            ax[i].title.set_text(subplot_name)
        subplot_name = 'Sum CC1-CC5'
        map = np.sum(array_map[current_idx, 1:, :, :], axis=0)
        map_range = np.percentile(map, [1.0, 99.0])
        im = ax[len(THRESHOLDS)].imshow(map, vmin=map_range[0], vmax=map_range[1])
        plt.colorbar(im, ax=ax[len(THRESHOLDS)], label='counts')
        ax[len(THRESHOLDS)].title.set_text(subplot_name)
        fig.suptitle(f'Total Counts - {tube_current} mA')
        fig.savefig(f'{fig_filepath}_stitchedmap_{tube_current}mA.png')
        plt.close()

        fig2 = plt.figure(figsize=(20.0, 15.0))
        sumCC_frames = np.sum(frames_data[current_idx, 1:, :, :], axis=0)
        plt.plot(range(FRAMES), sumCC_frames)
        plt.xlabel('frames')
        plt.ylabel('counts')
        plt.legend([f'asic {asic}' for asic in asics])

        fig2.tight_layout()
        fig2.suptitle(f'Sum CC1-CC5 Counts - {tube_current} mA')
        fig2.savefig(f'{fig_filepath}_frames_{tube_current}mA.png')
        plt.close()




def plot_threshold_sweep_visuals(sumCC_frames, thresh_list, fig_filepath):
    # sumCC_frames = np.zeros((num_sweep_dim, counters, FRAMES, len(fpga.selected_asics)), dtype='int')
    # For each tube current, plot raw count results
    fig2 = plt.figure(figsize=(20.0, 15.0))
    spectra = np.mean(np.sum(sumCC_frames, axis=-2), axis=-1)
    plt.plot(thresh_list, spectra)
    plt.xlabel('threshold value (AU)')
    plt.ylabel('counts')
    plt.legend([f'CC{cc_bin}' for cc_bin in range(spectra.shape[0])])

    fig2.tight_layout()
    fig2.suptitle('Threshold Sweep - Mean Counts')
    fig2.savefig(f'{fig_filepath}.png')
    plt.close()


def save_raw_data(fpga, asic_data_structs, output_path):
    for asic_idx in fpga.selected_asics:
        asic_serial = asic_serials[asic_idx]
        output_path_asic = os.path.join(output_path, asic_serial)
        sio.savemat(f'{output_path_asic}.mat', {'cc_struct': asic_data_structs[asic_idx]}, format='5',
                    do_compression=True)


def write_electrical_cal_asics(fpga, baseline):
    electrical_cal_start_time = time.time()
    for asic_idx, asic in fpga.iter_all_asics():
        asic_serial = asic_serials[asic_idx]

        electrical_cal_path = (fR'{CALIB_DIR}/{asic_serial}/'
                               fR'{electrical_cal_filename(baseline, asic_serial, asic.revision)}')

        logger.info('Writing electrical calibration for ASIC %d (%s) from %s', asic_idx, asic_serial,  # noqa: F821
                    electrical_cal_path)
        write_csv_electrical_cal(asic, electrical_cal_path)

    fpga.unmask_all_asics()
    logger.info('Electrical calibrations completed in %d s', time.time() - electrical_cal_start_time)  # noqa: F821


def write_energy_cal_asics(fpga, thresholds, baseline):
    if ENERGY_CAL:
        for asic_idx, asic in fpga.iter_selected_asics():
            asic_serial = asic_serials[asic_idx]

            energy_cal_dir = os.path.join(CALIB_DIR, asic_serial, 'energy')
            energy_cal_file = energy_cal_filename(BIAS, baseline['name'])

            logger.info('Writing energy calibration for ASIC %d (%s) from %s', asic_idx, asic_serial,
                        energy_cal_dir)
            write_csv_energy_cal(asic, os.path.join(energy_cal_dir, energy_cal_file), asic_serial, thresholds)
    else:
        for _, asic in fpga.iter_asics_by_version():
            # Use default energy calibration
            gain, offset = asic.threshold_gain_and_offset
            thresholds = [int(round((threshold - offset) / gain)) for threshold in thresholds]

            asic.write_thresh_chains(thresholds=thresholds)
    fpga.unmask_selected_asics()


def set_asic_baseline(fpga, baseline):
    # Set ASIC modes to current baseline settings
    for _, asic in fpga.iter_all_asics():
        asic.mode(**baseline['mode'])


def main():
    # If using ODMB
    detected = False
    while not detected:
        try:
            buffers = Buffer.detect(settings.BUFFER_IP_ADDRESSES)
            detected = True
        except (BufferError, BufferTimeoutError, socket.timeout):
            if easygui.ccbox(msg="Failed to detect buffer", title=" ", choices=("[T]ry again", "[C]ancel"),
                             image=None, default_choice='Try again', cancel_choice='Cancel'):
                pass
            else:
                break
    # Use the highest priority buffer to communicate with the DUT
    buffer = buffers[0]
    buffer.reset()

    # If Virtual Buffer
    # buffer = Virtual()
    # buffer.connect()

    # Connect instruments and fpga reset

    instruments = Instruments()
    fpga = CanonDm8XmedRevE(buffer, asics=asics_in_use)
    reset = False
    while not reset:
        try:
            fpga.reset()
            reset = True
        except (BufferError, BufferTimeoutError, RuntimeError):
            if eg.ccbox(msg="Failed to reset detector", title=" ", choices=("[T]ry again", "[C]ancel"),
                        image=None, default_choice='Try again', cancel_choice='Cancel'):
                pass
            else:
                break
    fpga.enable_debug = True
    fpga.select_counters(ec=COUNTERS['EC'], cc=COUNTERS['CC'], sec=COUNTERS['SEC'])

    # Set ASIC modes to current baseline settings
    # baseline = settings.DDC0864_BASELINES[settings.DDC0864_BASELINES]
    baseline = settings.baseline('B')
    set_asic_baseline(fpga, baseline)

    # Connect to HV based on detector type
    instruments.connect_hv(fpga)
    instruments.connect()
    instruments.shutter.put('open')

    # Load electrical calibrations for each ASIC
    write_electrical_cal_asics(fpga, baseline)

    # create output data template
    # asic_data_structs[asic_idx].data => (#num_sweep_dim, #counters, #frames, #row, #col)
    folder_name = eg.multenterbox(' Specify folder name or leave empty to use default folder name.',
                                  title='Folder Name', fields=['Folder Name'])
    if not folder_name[0]:
        folder_name = FOLDER_NAME[TEST_TYPE]
    output_path = os.path.join(WORKING_DIR, f'{SERIAL}_{folder_name[0]}')
    os.makedirs(output_path, exist_ok=True)

    try:
        # Enable status frame monitoring for Aurora-based detector modules
        if settings.ENABLE_MONITORING and isinstance(fpga, DmbFpga):
            fpga.monitor_status_frames = True
            fpga.status_frame_control(*settings.STATUS_FRAME_CONTROL)

        fpga.select_all_asics()

        if TEST_TYPE == 'XRAY_SWEEP':
            tube_currents = eg.multenterbox(msg='Enter tube currents as list. (e.g. [0.0, 3.0, 7.0])',
                                            title='Tube Current List', fields=['Tube current(mA)'])
            tube_currents = [float(i) for i in tube_currents[0].strip('][').split(',')]
            fig_filenames = eg.multenterbox(msg='Enter test notes for each capture/current. (e.g. slabA, slabB, slabC)',
                                            title='Test Notes',
                                            fields=[f'note {current} mA' for current in tube_currents])

            if tube_currents is None:
                raise KeyError

            # asic_data_structs[asic_idx].data => (#num_sweep_dim, #counters, #frames, #row, #col)
            write_energy_cal_asics(fpga, THRESHOLDS, baseline)
            asic_data_structs = prepare_output_data(fpga, num_sweep_dim=len(tube_currents), tube_currents=tube_currents)

            # for _, asic in fpga.iter_all_asics():
            #     asic.xray_capture_mode()

            instruments.hv.set_hv_voltage(1000)
            instruments.hv.set_hv_state('on')

            sweep_xray(fpga, asic_data_structs, tube_currents)
            plot_and_save_fig(fpga, asic_data_structs, TEST_TYPE, num_sweep_dim=len(tube_currents),
                              fig_filenames=fig_filenames, output_path=output_path, tube_currents=tube_currents)
            save_raw_data(fpga, asic_data_structs, output_path)


        elif TEST_TYPE == 'RISING_EDGE':
            tube_currents = eg.multenterbox(msg='Enter tube currents as list. (e.g. [0.0, 3.0, 7.0])',
                                            title='Tube Current List', fields=['Tube current(mA)'])
            tube_currents = [float(i) for i in tube_currents[0].strip('][').split(',')]
            fig_filenames = eg.multenterbox(msg='Enter test notes for each capture/current. (e.g. slabA, slabB, slabC)',
                                            title='Test Notes',
                                            fields=[f'note {current} mA' for current in tube_currents])

            if tube_currents is None:
                raise KeyError

            # asic_data_structs[asic_idx].data => (#num_sweep_dim, #counters, #frames, #row, #col)
            write_energy_cal_asics(fpga, THRESHOLDS, baseline)
            asic_data_structs = prepare_output_data(fpga, num_sweep_dim=len(tube_currents), tube_currents=tube_currents)

            instruments.hv.set_hv_voltage(1000)
            instruments.hv.set_hv_state('on')
            #instruments.shutter.put('pwm 1 1 10')

            #sweep_xray(fpga, asic_data_structs, tube_currents)
            rising_edge(fpga, asic_data_structs, tube_currents, instruments, SHUTTER_ON, SHUTTER_OFF, TOTAL_TIME)
            plot_and_save_fig(fpga, asic_data_structs, TEST_TYPE, num_sweep_dim=len(tube_currents),
                              fig_filenames=fig_filenames, output_path=output_path, tube_currents=tube_currents)
            save_raw_data(fpga, asic_data_structs, output_path)


        elif TEST_TYPE == 'THRESHOLD_SWEEP':
            start, end, step = eg.multenterbox(msg='Enter threshold start, end and step size. '
                                                   'Only integer values between 0 and 255.)',
                                               title='Threshold Settings', fields=['start', 'end', 'step'])
            sweep = range(int(start), (int(end) + 1) if int(end) <= MAX_THRESHOLD else (MAX_THRESHOLD + 1), int(step))

            tube_voltages = eg.multenterbox(msg='Enter tube voltages as list. (e.g. [100.0, 120.0, 160.0])',
                                            title='Tube Voltage List', fields=['Tube voltage(kVp)'])
            tube_voltages = [float(i) for i in tube_voltages[0].strip('][').split(',')]
            tube_currents = eg.multenterbox(msg='Enter tube currents as list. (e.g. [0.0, 3.0, 7.0])',
                                            title='Tube Current List', fields=['Tube current(mA)'])
            tube_currents = [float(i) for i in tube_currents[0].strip('][').split(',')]

            filter_list = eg.multenterbox(msg='Enter filters as list. (e.g. [3mmCu, 1mmCu, NoFilter])',
                                            title='Filter List', fields=['Filters'])
            filter_list = [i for i in filter_list[0].strip('][').split(',')]

            instruments.hv.set_hv_voltage(1000)
            instruments.hv.set_hv_state('on')

            for filter in filter_list:
                for tube_voltage in tube_voltages:
                    for tube_current in tube_currents:
                        output_path_curr = os.path.join(output_path, f'{tube_voltage}kVp_{tube_current}mA_{filter}')
                        os.makedirs(output_path_curr, exist_ok=True)
                        # asic_data_structs[asic_idx].data => (#num_sweep_dim, #counters, #frames, #row, #col)
                        asic_data_structs = prepare_output_data(fpga, num_sweep_dim=len(sweep),
                                                                tube_currents=tube_current,
                                                                tube_voltages=tube_voltage, filters=filter,
                                                                threshold_params=[int(start), int(end), int(step)])
                        for _, asic in fpga.iter_asics_by_version():
                            asic.thresh_capture_mode()

                        eg.msgbox(msg=f'Compelete set up for {tube_voltage} kVp {tube_current} mA {filter}.')
                        sweep_thresholds(fpga, asic_data_structs, sweep, step=int(step))
                        plot_and_save_fig(fpga, asic_data_structs, TEST_TYPE, num_sweep_dim=len(sweep),
                                          fig_filenames=folder_name[0], output_path=output_path_curr, thresh_list=sweep)

                        # save raw_data
                        save_raw_data(fpga, asic_data_structs, output_path_curr)
                        print(f'data saved for {output_path_curr}')

    except (Exception, KeyboardInterrupt):
        logger.exception('Unhandled exception')

    finally:
        if instruments.hv:
            instruments.hv.set_hv_state('off')
        instruments.disconnect()


if __name__ == '__main__':
    pcct.config_logging(console_level=logging.DEBUG, file_level=None)
    pcct.load_packages()
    main()
