'''
SPIM App
*******************
@authors: Emma Martinelli, Andrea Bassi. Politecnico di Milano

'''
from ScopeFoundry import BaseMicroscopeApp

def add_path(path):
    import sys
    import os
    # add path to ospath list, assuming that the path is in a sybling folder
    from os.path import dirname
    sys.path.append(os.path.abspath(os.path.join(dirname(dirname(__file__)),path)))


#add your Pi model and serial number
SERIAL = {'M-403.4DG': '0115500028',  # SPIM translator stage
          'V-524.1AA': '0119024343',  # voice coil
          }

class SPIM_app(BaseMicroscopeApp):
    name = 'SPIM_App'
    
    def setup(self):
        
        #Add hardware components

        print("Adding Camera Hardware Components")
        add_path('NAC_ScopeFoundry')
        from NAC_hw import NeoAndorHW
        self.add_hardware(NeoAndorHW(self))

        print("Adding Shutter Hardware Components")
        add_path('Shutter_ScopeFoundry')
        from shutter_hw import ShutterHW
        self.add_hardware(ShutterHW(self))

        print("Adding Translator Hardware Components")
        add_path('PI_ScopeFoundry')
        # from PI_CG_hardware import PI_CG_HW
        # self.add_hardware(PI_CG_HW(self, serial='0115500028'))
        from PI_hardware import PI_HW
        self.add_hardware(PI_HW(self, serial='0119024343'))   #voice coil

        # Add measurement components
        print("Create Measurement objects")
        from SPIM_measure import SpimMeasure
        self.add_measurement(SpimMeasure(self))

if __name__ == '__main__':
    import sys
    import os
    app = SPIM_app(sys.argv)

    # current file dir and select settings file:
    # path = os.path.dirname(os.path.realpath(__file__))
    # new_path = os.path.join(path, 'Settings', 'Settings.ini')
    # print(new_path)
    #
    # app.settings_load_ini(new_path)
    # # connect all the hardwares
    # for hc_name, hc in app.hardware.items():
    #     hc.settings['connected'] = True

    sys.exit(app.exec_())