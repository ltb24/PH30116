"""
Supporting code for PH30116 - Data analysis and research methods in observational astrophysics - University of Bath
ImageSimulator contains code to simulate images witn varying sky levels to teach photometry
"""

import numpy
import os
import pylab
from MyExceptions import Hell, TheDead, Hope, InputError, StupidError, Cthulhu

class SimuIma:
    """
    This class simulates astronomical images, including adding psfs, background, noise.
    Created images allow to practice photometry.
    """
    def __init__(self, size=(400, 600)):
        """
        :param size: size of the created image
        """
        self._ima = numpy.zeros(size) #Initialize image
        self._size = size
        #needed for calculations.
        self._x = numpy.tile(numpy.arange(0, size[1]), (size[0], 1))
        self._y = numpy.swapaxes(numpy.tile(numpy.arange(0, size[0]), (size[1], 1)), 0, 1)
        #this will be the noise image
        self._realima = numpy.zeros(size)
        #tracking changes and locking
        self._history = []
        self._lock = False
        self._practicemode = False
        self._practicedict = {}

    def add_bg(self, level):
        """
        Add a background level
        :param level: background level in cts/pixel
        :return:
        """
        if self._lock:
            raise InputError('Oi! This image has been locked.')
        else:
            self._ima += level
            self._history.append('Background added level  = %s' %level)

    def addPSF(self, x, y, sigma, intflux=1.):
        """
        Add a Gaussian PSF
        :param x: x position
        :param y: y position
        :param sigma: width of PSF (sigma), same in x and y
        :param intflux: integrated flux of psf
        :return:
        """
        if self._lock:
            raise InputError('Oi! This image has been locked.')
        else:
            self._ima += (1/(2*numpy.pi*(sigma**2)))*\
                         numpy.exp(- ((self._x - x) ** 2 + (self._y - y) ** 2) / (2* (sigma ** 2)))\
                         * intflux
            self._history.append('PSF added x = %s y = %s sigma = %s intflux = %s' %(x, y, sigma, intflux))

    def add_shot(self, scale):
        """
        Add shot (Poisson) noise
        :param scale: scaling applied before calculating shot noise
        :return:
        """
        if self._lock:
            raise InputError('Oi! This image has been locked.')
        else:
            self._realima = numpy.random.poisson(self._ima*scale)
            self._history.append('Shot noise added scale = %s' %scale)

    def add_ron(self, std):
        """
        Add read out noise
        :param std: expected read out nosie per pixel
        :return:
        """
        if self._lock:
            raise InputError('Oi! This image has been locked.')
        else:
            self._realima += numpy.random.poisson(std, size=self._size)
            self._history.append('RON added std = %s' % std)

    def write(self, filename, raw=False):
        """
        Dumps the current simulated data to a fist file.
        @param filename: name of output file
        @type filename: string
        """
        if os.path.isfile(filename):
            print('File %s already exists.' % filename)
            print('Choose another filename and run WriteTo to save.')
            print('Now exiting.')
            return
        if raw:
            tmp_object = pyfits.PrimaryHDU(self._ima)
        else:
            tmp_object = pyfits.PrimaryHDU(self._realima)
        tmp_object.writeto(filename)
        self._history.append('File written to %s raw = %s' %(filename, raw))

    def show_raw(self):
        """
        Show the non-noisy image.
        :return:
        """
        pylab.imshow(self._ima)
        pylab.colorbar()
        pylab.ylim(0, self._size[0])

    def show_ima(self):
        """
        Show the noisy image.
        :return:
        """
        pylab.imshow(self._realima)
        pylab.colorbar()
        pylab.ylim(0, self._size[0])

    def plot_x(self, n, raw=False):
        """
        Plot a row of the image
        :param n:
        :return:
        """
        if raw:
            pylab.plot(self._ima[:, n])
        else:
            pylab.plot(self._realima[:, n])

    def plot_y(self, n, raw=False):
        """
        Plot a column
        :param n:
        :param raw:
        :return:
        """
        if raw:
            pylab.plot(self._ima[n])
        else:
            pylab.plot(self._realima[n])


    def reset(self):
        """
        Reset the image
        :return:
        """
        if self._lock:
            raise InputError('Oi! This image has been locked.')
        else:
            self._ima = numpy.zeros(self._size)
            self._realima = numpy.zeros(self._size)
            self._history = []

    def get_data(self):
        return self._realima


    def practiceima(self, npsf=2, psffluxrange=[500, 1000], bgrange=[2, 10], sigmarange=[3, 6], ronrange=[1, 10],
                    shot=True, ron=True, edge=0.1):
        """
        Create a random practice image.
        :param npsf: number of objects added
        :param psffluxrange: range in which psf flux will be randomly chosen
        :param bgrange: range for background
        :param sigmarange: range for FWHM, will be idential for all objects
        :param ronrange: range in which RON will be chosen
        :param shot: add shot noise? (Boolean)
        :param ron: add RON? (bolean)
        :param edge: edge of frame left out of placement of PSFs
        :return:
        """
        #Need to add sanity checks for the inputs
        if type(npsf) is not int or npsf <=0:
            raise InputError("npsf needs to be an integer >= 0")
        if len(psffluxrange) != 2:
            raise InputError('psffluxrange needs to be of length 2')
        if len(bgrange) != 2:
            raise InputError('bgrange needs to be of length 2')
        if len(sigmarange) != 2:
            raise InputError('sigmarange needs to be of length 2')
        if len(ronrange) != 2:
            raise InputError('ronrange needs to be of length 2')
        if type(shot) is not bool:
            raise InputError("shot needs to be boolean")
        if type(ron) is not bool:
            raise InputError("shot needs to be boolean")
        if edge < 0 or edge > 0.5:
            raise InputError("edge needs to be between 0 and 0.5")
        #Resets and unlocks the image
        self._lock = False
        self.reset()
        self._practicedict = {}
        #Add psf
        #add PSF info to dictionary
        self._practicedict['npsfs'] = npsf
        self._practicedict['psf_x'] = []
        self._practicedict['psf_y'] = []
        self._practicedict['psf_flux'] = []
        #get FWHM value and store
        sigma = numpy.random.uniform(sigmarange[0], sigmarange[1], 1)[0]
        self._practicedict['sigma'] = sigma
        #add PSFs, looping
        for i in range(npsf):
            #create random values for instance of this PSF
            x = numpy.random.uniform(edge*self._size[0], (1-edge)*self._size[0], 1)[0]
            y = numpy.random.uniform(edge*self._size[1], (1-edge)*self._size[1], 1)[0]
            flux = numpy.random.uniform(psffluxrange[0], psffluxrange[1], 1)[0]
            #add PSF
            self.addPSF(x, y, sigma, flux)
            #store PSF
            self._practicedict['psf_x'].append(x)
            self._practicedict['psf_y'].append(y)
            self._practicedict['psf_flux'].append(flux)
        #add bg
        bg = numpy.random.uniform(bgrange[0], bgrange[1], 1)[0]
        self._practicedict['bg'] = bg
        self.add_bg(bg)
        #add noise
        #RON
        if ron:
            ron = numpy.random.uniform(ronrange[0], ronrange[1], 1)[0]
            self._practicedict['ron'] = ron
            self._practicedict['ronflag'] = ron
            self.add_ron(ron)
        else:
            self._practicedict['ronflag'] = ron
        #add shot noise, if requested
        if shot:
            self._practicedict['shotflag'] = shot
            self.add_shot(1)
        else:
            self._practicedict['shotflag'] = shot
        #locking the image
        self._practicemode = True
        self._lock = True
        #showing the image
        self.show_ima()


    def explain_practiceima(self):
        """
        Prints out information about the practice image
        :return:
        """
        if self._practicemode is False:
            raise StupidError("Oi! you don't have a practice image for me to explain.")
        else:
            print("There are %i Objects in this image" % self._practicedict['npsfs'])
            print("They have the following parameters:")
            print("x, y, flux")
            print("-----------")
            for i in range(self._practicedict['npsfs']):
                print('%.2f, %.2f, %.2f' %(self._practicedict['psf_x'][i], self._practicedict['psf_y'][i],
                                           self._practicedict['psf_flux'][i]))
            print("-----------")
            print("Background level: %.2f" % self._practicedict['bg'])
            print("Radius/FWHM: %.2f/%.2f" % (self._practicedict['sigma'], 2.35*self._practicedict['sigma']))
            if self._practicedict['ronflag']:
                print("RON: %.2f added" % self._practicedict['ron'])
            else:
                print("No RON added")

            if self._practicedict['shotflag']:
                print("Shot noise added")
            else:
                print("No Shot noise added")

    def practiceima_positions(self):
        """
        Prints out information about the practice image
        :return:
        """
        if self._practicemode is False:
            raise StupidError("Oi! you don't have a practice image for me to explain.")
        else:
            print("There are %i Objects in this image with positions" % self._practicedict['npsfs'])
            print("x, y")
            print("-----------")
            for i in range(self._practicedict['npsfs']):
                print('%.2f, %.2f' %(self._practicedict['psf_x'][i], self._practicedict['psf_y'][i]))

    def guess_psf(self, x, y, flux, error, poserr):
        """
        Compare a guess of the psf properties with the results.
        Matches will by accepted within actual value +- 1 error
        :param x: guess for x
        :param y: guess for y
        :param flux: guess for flux
        :param error: error in flux guess
        :param poserr: error in position
        :return:
        """
        if self._practicemode is False:
            raise StupidError("Oi! you don't have a practice image to make a guess for.")
        match = False
        print("your guess: x,y = %.2f, %.2f (+- %.2f), flux=%.2f +- %.2f" % (x, y, poserr, flux, error))
        for i in range(self._practicedict['npsfs']):
            xpsf, ypsf, fpsf = self._practicedict['psf_x'][i], self._practicedict['psf_y'][i],\
                               self._practicedict['psf_flux'][i]
            if xpsf - poserr < x  and  xpsf + poserr > x:
                print('Matched in X position')
                if ypsf - poserr < y and ypsf + poserr > y:
                    print('Matched in Y position')
                    if fpsf - error < flux and fpsf + error > flux:
                        print('Matching psf with x = %.2f, y = %.2f, flux = %.2f' %(xpsf, ypsf, fpsf))
                        print('Well done!')
                        match = True
                    else:
                        print("Match in position, but not flux.")
        if match is False:
            print('No match found.')

    def lock(self):
        """
        Lock the image.
        :return:
        """
        self._lock = True

    def unlock(self):
        """
        Unlock the image
        :return:
        """
        self._lock = False




class centred_psf_highSN(SimuIma):
    def __init__(self, size=(50, 50)):
        """
        :param size:
        """
        SimuIma.__init__(self, size=size)
        self.practiceima(npsf=1, edge=0.45, psffluxrange=[5000, 50000], bgrange=[1,5], sigmarange=[3,6], ronrange=[1, 5])
        print("This class will simulate a single PSF in the centre of the field with high SN. RON is %.2f" % self._practicedict['ron'])


class centred_psf_lowSN(SimuIma):
    def __init__(self, size=(50, 50)):
        """
        :param size:
        """
        SimuIma.__init__(self, size=size)
        self.practiceima(npsf=1, edge=0.45, psffluxrange=[500, 2000], bgrange=[1,5], sigmarange=[3,6], ronrange=[1, 5])
        print("This class will simulate a single PSF in the centre of the field with low SN. RON is %.2f" % self._practicedict['ron'])


class crowded_field(SimuIma):
    def __init__(self, size=(100, 100)):
        SimuIma.__init__(self, size=size)
        self.npsf = numpy.random.randint(5, 10)
        self.practiceima(npsf=self.npsf, edge=0.05, psffluxrange=[4000, 10000], bgrange=[1, 5], sigmarange=[3, 6], ronrange=[1, 5])
        print("A crowded field with several psfs. RON is %.2f" % self._practicedict['ron'])
    def show_objectnumber(self):
        print("There are %i PSFs in this field. Good hunting!" % self.npsf)
    def show_positions(self):
        self.practiceima_positions()

class tutorial_image(SimuIma):
    """
    The tutorial image, hidden from the students, he he he.
    """
    def __init__(self):
        SimuIma.__init__(self, size=(50,50))
        self.addPSF(20, 20, 2, 5000)
        self.addPSF(30, 30, 2, 400)
        self.add_bg(20)
        self.add_shot(1)
        self.lock()

class calibrate_object(SimuIma):
    """
    Simulates a standard star and object with random sky background random fluxes in both sources
    """
    def __init__(self, size=(75, 75)):
        SimuIma.__init__(self, size=size)
        self.practiceima(npsf=2, edge=0.25, psffluxrange=[5000, 10000], bgrange=[1, 5], sigmarange=[3, 6], ronrange=[1, 5])
        self.explain_calib()
        raise Hope("this needs scaling")

    def explain_calib(self):
        """
        Prints out information about the practice image
        :return:
        """
        if self._practicemode is False:
            raise StupidError("Oi! you don't have a practice image for me to explain.")
        else:
            print("The standard star information is:")
            print("x, y, flux")
            print("-----------")
            print('%.2f, %.2f, %.2f' % (self._practicedict['psf_x'][0], self._practicedict['psf_y'][0],
                                            self._practicedict['psf_flux'][0]))
            print("-----------")
            if self._practicedict['ronflag']:
                print("RON: %.2f added" % self._practicedict['ron'])
            else:
                print("No RON added")

            if self._practicedict['shotflag']:
                print("Shot noise added")
            else:
                print("No Shot noise added")






