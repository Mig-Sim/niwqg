# Generic methods for writing to disk

import os
import h5py
import cupy as cp
def initialize_save_snapshots(self,path):

    """ Initializes class variables for saving snapshots.
        Sets the path and creates directory if needed.

        Parameters
        ----------
        path:  string (required)
                    Location to save model outputs.
    """

    self.fno = path

    if (not os.path.isdir(self.fno)) & self.save_to_disk:
        os.makedirs(self.fno)
        os.makedirs(self.fno+"/snapshots/")

def file_exist(fno,overwrite=True):

    """ Check whether file exists.

        Parameters
        ----------
        overwrite:  string (optional)
                        If True, then overwrite extant files.
    """

    if os.path.exists(fno):
        if overwrite:
            os.remove(fno)
        else: raise IOError("File exists: {0}".format(fno))

def save_setup(self,):

    """ Save set up of model simulations.
    """

    if self.save_to_disk:

        fno = self.fno + '/setup.h5'

        file_exist(fno,overwrite=self.overwrite)
        if self.use_cuda:
            import cupy as cp

            x = cp.asnumpy(self.x)
            y = cp.asnumpy(self.y)
            wv = cp.asnumpy(self.wv)
            kk = cp.asnumpy(self.kk)
            ll = cp.asnumpy(self.ll)
        else:
            x = self.x
            y = self.y
            wv = self.wv
            kk = self.kk
            ll = self.ll
        h5file = h5py.File(fno, 'w')
        h5file.create_dataset("grid/nx", data=(self.nx),dtype=int)
        h5file.create_dataset("grid/x", data=(x))
        h5file.create_dataset("grid/y", data=(y))
        h5file.create_dataset("grid/wv", data=(wv))
        h5file.create_dataset("grid/k", data=(kk))
        h5file.create_dataset("grid/l", data=(ll))
        # h5file.create_dataset("constants/f0", data=(self.f))
        h5file.close()

def save_snapshots(self, fields=['t','q','p']):

    """ Save snapshots of model simulations.

        Parameters
        ----------
        fields:  list of strings (optional)
                    The fields to save. Default is time ('t'), potential vorcitiy ('q')
                                        and streamfunction ('p').
    """

    if ( ( not (self.tc%self.tsnaps) ) & (self.save_to_disk) ):

        fno = self.fno + '/snapshots/{:015.0f}'.format(self.t)+'.h5'

        file_exist(fno)

        h5file = h5py.File(fno, 'w')

        for field in fields:
            if field == 't':
                data = self.t
                h5file.create_dataset(field, data=data)
            else:
                data = getattr(self, field)
                if self.use_cuda:
                    data = cp.asnumpy(data)
                h5file.create_dataset(field, data=data)

        h5file.close()
    else:
        pass

def save_diagnostics(self):

    """ Save diagnostics of model simulations.
    """
    fno = self.fno + '/diagnostics.h5'

    file_exist(fno,overwrite=self.overwrite)

    h5file = h5py.File(fno, 'w')

    for key in self.diagnostics.keys():

        data = self.diagnostics[key]['value']

        if self.use_cuda and isinstance(data, cp.ndarray):
            data = cp.asnumpy(data)

        h5file.create_dataset(key, data=data)

    h5file.close()
