import numpy as np

# this class is meant to store the min and max images, which will get
# updated
class HiLo():
    def __init__(self):
        self.sidelen = 2048 # square ; this is in coadd pixel space
        self.im_min = np.zeros((self.sidelen, self.sidelen), dtype=float) + 1.0e30
        self.im_max = np.zeros((self.sidelen, self.sidelen), dtype=float) - 1.0e30
        self.id_min = np.zeros((self.sidelen, self.sidelen), dtype='S9')
        self.id_max = np.zeros((self.sidelen, self.sidelen), dtype='S9')

    def update(self, rimg):

        id = rimg.scan_id + str(rimg.frame_num).zfill(3)

        x0_coadd = rimg.coextent[0]
        x1_coadd = rimg.coextent[1] + 1
        y0_coadd = rimg.coextent[2]
        y1_coadd = rimg.coextent[3] + 1

        slc = slice(y0_coadd, y1_coadd), slice(x0_coadd, x1_coadd)

        sub_im_min = self.im_min[slc]
        sub_im_max = self.im_max[slc]

        sub_id_min = self.id_min[slc]
        sub_id_max = self.id_max[slc]

        iy_min, ix_min = np.where((rimg.rimg < sub_im_min))
        iy_max, ix_max = np.where((rimg.rimg > sub_im_max))

        # if statements !! if len(iy_min) > 0 ... etc.

        sub_im_min[iy_min, ix_min] = (rimg.rimg)[iy_min, ix_min]
        sub_im_max[iy_max, ix_max] = (rimg.rimg)[iy_max, ix_max]

        self.im_min[slc] = sub_im_min
        self.im_max[slc] = sub_im_max

        sub_id_min[iy_min, ix_min] = id
        sub_id_max[iy_max, ix_max] = id

        self.id_min[slc] = sub_id_min
        self.id_max[slc] = sub_id_max
