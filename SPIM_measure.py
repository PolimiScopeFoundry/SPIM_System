'''
Neao Andor Camera Measurement
'''
from ScopeFoundry import Measurement
from ScopeFoundry.helper_funcs import sibling_path, load_qt_ui_file
from ScopeFoundry import h5_io
import pyqtgraph as pg
import numpy as np
import os, time
import matplotlib.pyplot as plt
import h5py

class SpimMeasure(Measurement):
    name = "SPIM_measure"

    def setup(self):
        self.ui_filename = sibling_path(__file__, "camera_with_mip.ui")
        self.ui = load_qt_ui_file(self.ui_filename)

        self.settings.New('save_h5', dtype=bool, initial=False)

        self.settings.New('Z_step', dtype=int, unit='um', initial=10)
        self.settings.New('Z_series', dtype=int, initial=10)
        self.settings.New('Time_series', dtype=int, initial=1)

        self.settings.New('Measurement time', dtype=float, unit='s', initial=0, ro=False)
        # self.settings.New('Acquisition time', dtype=float, unit='s',
        #                   initial=self.settings['Z_series'] *  self.image_gen.exposure_time,
        #                   ro=True)

        # how often we want to update the display
        self.settings.New('refresh_period', dtype=float, unit='s', spinbox_decimals=3, initial=0.05, vmin=0)

        #RISOLUZIONE: dimensione del pixel della camera e sella profondità in z
        self.settings.New('xsampling', dtype=float, unit='um', initial=0.65, ro=False)
        self.settings.New('ysampling', dtype=float, unit='um', initial=0.65, ro=False)
        self.settings.New('zsampling', dtype=float, unit='um', initial=10, ro=False)

        self.auto_range = self.settings.New('auto_range', dtype=bool, initial=True)
        self.settings.New('auto_levels', dtype=bool, initial=True)
        self.settings.New('level_min', dtype=int, initial=60)
        self.settings.New('level_max', dtype=int, initial=4000)

        self.mip_type=self.settings.New(name='mip_type',dtype=str, choices=['mean', 'max'], initial = 'max', ro=False)
        self.save_type=self.settings.New(name='save_type',dtype=str, choices=['stack','mip','all'], initial = 'mip', ro=False)
    

        '''AGGIUNGI UN BOTTONE:
        self.add_operation('name',op_func)
        example: self.add_operation('measure',self.measure)'''

        self.image_gen = self.app.hardware['NeoAndorHW']
        self.stage = self.app.hardware['PI_CG_HW']
        self.shutter_measure = self.app.hardware['Shutter']


    def setup_figure(self):
        """
        Runs once during App initialization, after setup()
        This is the place to make all graphical interface initializations,
        build plots, etc.
        """
        # connect ui widgets to measurement/hardware settings or functions
        self.ui.start_pushButton.clicked.connect(self.start)
        self.ui.interrupt_pushButton.clicked.connect(self.interrupt)
        self.settings.save_h5.connect_to_widget(self.ui.save_h5_checkBox)
        
        self.settings.auto_levels.connect_to_widget(self.ui.autoLevels_checkbox)
        self.settings.auto_range.connect_to_widget(self.ui.autoRange_checkbox)
        self.settings.level_min.connect_to_widget(self.ui.min_doubleSpinBox)
        self.settings.level_max.connect_to_widget(self.ui.max_doubleSpinBox)


        self.settings.mip_type.connect_to_widget(self.ui.mip_selector)
        self.settings.save_type.connect_to_widget(self.ui.save_selector)

        #shutter add checkbox
        self.shutter_measure.shutter_closed.connect_to_widget(self.ui.shutter_checkbox)

        # Set up pyqtgraph graph_layout in the UI
        self.imv = pg.ImageView()
        self.ui.imageLayout.addWidget(self.imv)
        colors = [(0, 0, 0),
                  (45, 5, 61),
                  (84, 42, 55),
                  (150, 87, 60),
                  (208, 171, 141),
                  (255, 255, 255)
                  ]
        cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
        self.imv.setColorMap(cmap)

        self.mip_imv = pg.ImageView()
        self.ui.mipLayout.addWidget(self.mip_imv)
        colors = [(0, 0, 0),
                  (45, 5, 61),
                  (84, 42, 55),
                  (150, 87, 60),
                  (208, 171, 141),
                  (255, 255, 255)
                  ]
        cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
        self.mip_imv.setColorMap(cmap)

    def update_display(self):
        """
        Displays (plots) the numpy array self.buffer.
        This function runs repeatedly and automatically during the measurement run.
        its update frequency is defined by self.display_update_period
        """
        self.image_gen.read_from_hardware()
        self.stage.read_from_hardware()

        self.display_update_period = self.settings['refresh_period']

        # self.settings['progress'] = (self.frame_index + 1) * 100 / self.length
        self.settings['progress'] = self.frame_index * 100 / self.length

        if hasattr(self, 'img'):
            # show the image
            self.imv.setImage(self.img,               #self.img.T,
                              autoLevels=self.settings['auto_levels'],
                              autoRange=self.auto_range.val,
                              levelMode='mono'
                              )

            if self.settings['auto_levels']:
                lmin, lmax = self.imv.getHistogramWidget().getLevels()
                self.settings['level_min'] = lmin
                self.settings['level_max'] = lmax
            else:
                self.imv.setLevels(min=self.settings['level_min'],
                                   max=self.settings['level_max'])
                
        if hasattr(self, 'meanIP_img') or hasattr(self, 'max_img'):
            if self.settings['mip_type'] == 'max': 
                mip_to_show = self.maxIP_img
            elif self.settings['mip_type'] == 'mean': 
                mip_to_show = self.meanIP_img

            self.mip_imv.setImage(mip_to_show,
                              autoLevels=True,
                              autoRange=self.auto_range.val,
                              levelMode='mono'
                              )

    def measure(self):
        self.stage.read_from_hardware()
        self.image_gen.read_from_hardware()
        self.shutter_measure.read_from_hardware()

        self.stage.motor.go_home()
        self.stage.motor.wait_on_target()

        # print('translator start: ', self.stage.motor.get_position())

        step_length = self.settings['Z_step'] * 0.001
        self.length_saving = space_frame = self.settings['Z_series']
        time_frame = self.settings['Time_series']
        measure_time = self.settings['Measurement time']

        start = 0
        stop = step_length * space_frame
        time_exp = self.image_gen.exposure_time.value
        print('start ', start)
        print('stop ', stop)
        print('time expo ', time_exp)
        # compute the velocity
        velocity = self.stage.motor.PI_velocity(time_exp, step_length)

        hstart, hend, vstart, vend, hbin, vbin = self.image_gen.camera.roi_get()
        w = hend - hstart
        h = vend - vstart

        self.length = num_frame = space_frame * time_frame

        self.create_h5_file_ext(time_frame)

        self.image_gen.camera.acquisition_setup(num_frame)
        self.image_gen.camera.acquisition_start()

        self.frame_index = 0
        time_tot = np.zeros(shape=(time_frame, 1))
        self.shutter_measure.shutter.open_shutter()
        for time_idx in range(0, time_frame):
            self.stage.motor.wait_on_target()
            t0 = time.perf_counter()
            # print('initial position: ', self.stage.motor.get_position())
            self.stage.motor.set_velocity(velocity)
            self.stage.motor.move_absolute(stop)

            maxIP_img = np.zeros((h,w))
            meanIP_img = np.zeros((h,w))


            for frame_idx_ext in range(0, space_frame):
                t1 = time.perf_counter()
                self.frame_index += 1
                self.image_gen.camera.image_wait()
                img = self.image_gen.camera.image_read()

                # time_acq[frame_idx_ext,0] = time.perf_counter() - t1
                # print('acquisition time: ', t_acq)
                # t2 = time.perf_counter()

                maxIP_img = np.maximum(maxIP_img, img) 
                meanIP_img += img/space_frame

                # time_save[frame_idx_ext, 0] = time.perf_counter() - t2
                # print('saving timing: ', time_save[frame_idx_ext, 0])

                self.maxIP_img = maxIP_img
                self.meanIP = meanIP_img
                self.img = img

                if self.settings['save_type']=='stack' or self.settings['save_type']=='all':
                    # access the group and save the image
                    t_group = self.h5_group[f't{time_idx}']
                    self.image_h5_ext = t_group['c0/image']
                    self.image_h5_ext[frame_idx_ext, :, :] = img.T
                    self.h5file.flush()

                if self.interrupt_measurement_called:
                    self.shutter_measure.shutter.close_shutter()
                    break

            if self.settings['save_type']=='mip' or self.settings['save_type']=='all':
                MIP_group = self.h5_group['mip']
                self.image_mip_max = MIP_group['c0/MIP_max']
                self.image_mip_max[time_idx,:,:] = maxIP_img.T
                self.image_mip_mean[time_idx,:,:] = meanIP_img.T
                self.h5file.flush()


            self.stage.motor.set_velocity(2.5)
            self.stage.motor.move_absolute(start)

            time_tot[time_idx, 0] = time.perf_counter() - t0
            # print('time tot: ', time_tot)
            # print('measurement time: ', measure_time)

            # in case you want to do a timelapse
            if measure_time > time_tot[time_idx, 0]:
                self.shutter_measure.shutter.close_shutter()
                wait_frame = measure_time-time_tot[time_idx, 0]
                time.sleep(wait_frame)
                self.shutter_measure.shutter.open_shutter()

        x = np.arange(num_frame)
        # print(time_acq)

        # print('mean delay: ', np.mean(time_save))
        print('time tot: ', time_tot)
        print('mean single stacK: ', np.mean(time_tot))
        # print('mean camera time ', np.mean(time_acq[:,0]))
        # print('mean saving time', np.mean(time_acq[:,1]))
        self.image_gen.camera.acquisition_stop()
        self.shutter_measure.shutter.close_shutter()

        # make sure to close the data file
        self.h5file.close()
        self.settings['save_h5'] = False
        # print('total acquisition time ', time.perf_counter()-t)

    def measure_ext(self):
        self.stage.read_from_hardware()
        self.image_gen.read_from_hardware()

        self.stage.motor.go_home()
        self.stage.motor.wait_on_target()

        print('start: ', self.stage.motor.get_position())

        step_length = self.settings['Z_step'] * 0.001
        self.length_saving = space_frame = self.settings['Z_series']
        time_frame = self.settings['Time_series']

        if not self.speed_check(step_length):
            return

        self.length = num_frame = space_frame * time_frame

        self.create_h5_file_ext(time_frame)

        self.image_gen.camera.acquisition_setup(num_frame)
        self.image_gen.camera.acquisition_start()

        self.frame_index = 0
        t = time.perf_counter()
        for time_idx in range(time_frame):
            if time_idx % 2 == 0:
                t_end = step_length*space_frame+0.5
                self.stage.motor.trigger(step_length, t_end)
            else:
                t_end = -0.5
                self.stage.motor.trigger(-step_length, t_end)

            for frame_idx in range(space_frame):
                t1 = time.perf_counter()
                self.frame_index += 1
                if time_idx % 2 == 0:
                    self.frame_index_saving = frame_idx
                else:
                    self.frame_index_saving = space_frame - frame_idx - 1

                print(self.frame_index_saving)

                self.image_gen.camera.image_wait()
                self.img = self.image_gen.camera.image_read()

                t_acqui = time.perf_counter() - t1
                t_a_1 = time.perf_counter()
                print('acquisition time: ', t_acqui)

                # access the group and save the image
                t_group = self.h5_group[f't{time_idx}']
                image_h5 = t_group['c0/image']
                image_h5[self.frame_index_saving, :, :] = self.img
                self.h5file.flush()

                print('saving time: ', time.perf_counter() - t_a_1)

                if self.interrupt_measurement_called:
                    break
        self.image_gen.camera.acquisition_stop()

        # make sure to close the data file
        self.h5file.close()
        self.settings['save_h5'] = False
        print(time.perf_counter()-t)

    def run(self):

        try:
            #start the camera
            self.frame_index = -1

            self.image_gen.read_from_hardware()
            self.length = self.image_gen.frame_num.val

            self.image_gen.camera.acquisition_clear()
            self.image_gen.camera.acquisition_setup(self.image_gen.frame_num.val)
            self.image_gen.camera.acquisition_start()

            # continuously get the last frame and put it in self.image, in order to
            # show it via self.update_display()

            while not self.interrupt_measurement_called:

                # If measurement is called, stop the acquisition, call self.measure
                # and get out of run()
                if self.settings['save_h5']:
                    # measure is triggered by save_h5 button
                    self.image_gen.camera.acquisition_stop()
                    if self.image_gen.settings['trigger'] == 'int':
                        self.measure()
                    elif self.image_gen.settings['trigger'] == 'ext':
                        self.measure_ext()
                    break

                self.image_gen.camera.image_wait()
                self.img = self.image_gen.camera.image_read()

                if self.interrupt_measurement_called:
                    break

        finally:
            self.image_gen.camera.acquisition_stop()
            # self.image_gen.settings['trigger'] = 'int'
            if self.settings['save_h5'] and hasattr(self, 'h5file'):
                # make sure to close the data file
                self.h5file.close()
                self.settings['save_h5'] = False

    def create_saving_directory(self):

        if not os.path.isdir(self.app.settings['save_dir']):
            os.makedirs(self.app.settings['save_dir'])

    def create_h5_file_ext(self, t_frame):
        self.create_saving_directory()
        # file name creation
        timestamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
        sample = self.app.settings['sample']
        # sample_name = f'{timestamp}_{self.name}_{sample}.h5'
        if sample == '':
            sample_name = '_'.join([timestamp, self.name])
        else:
            sample_name = '_'.join([timestamp, self.name, sample])

        self.fname = os.path.join(self.app.settings['save_dir'], sample_name + '.h5')

        self.h5file = h5_io.h5_base_file(app=self.app, measurement=self, fname=self.fname)
        self.h5_group = h5_io.h5_create_measurement_group(measurement=self, h5group=self.h5file)

        img_size = list(self.image_gen.camera.image_size())
        # terrible but there is no other way as the image is created later
        dtype = 'uint16'

        if self.settings['save_type']=='stack' or self.settings['save_type']=='all':
            length = self.length_saving
            for t_idx in range (t_frame):
                # Group creation for each t_index
                t_group = self.h5_group.create_group(f't{t_idx}')

                self.image_h5_ext = t_group.create_dataset(name='c0/image',
                                                    shape=[length, img_size[0], img_size[1]],
                                                    dtype=dtype)
                self.image_h5_ext.attrs['element_size_um'] = [self.settings['zsampling'], self.settings['ysampling'],
                                                        self.settings['xsampling']]

        if self.settings['save_type']=='mip' or self.settings['save_type']=='all':
            mip_group = self.h5_group.create_group('mip')
            #no possibilities to save the time as attr
            self.image_mip_max = mip_group.create_dataset(name='c0/MIP_max',
                                                        shape=[t_frame, img_size[0], img_size[1]],
                                                        dtype=dtype)
            self.image_mip_max.attrs['element_size_um'] = [0, self.settings['ysampling'],
                                                        self.settings['xsampling']]
            self.image_mip_mean = mip_group.create_dataset(name='c0/MIP_mean',
                                                        shape=[t_frame,img_size[0], img_size[1]],
                                                        dtype=dtype)
            self.image_mip_mean.attrs['element_size_um'] = [0, self.settings['ysampling'],
                                                      self.settings['xsampling']]
