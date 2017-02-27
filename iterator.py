import os
import numpy as np
import PIL.Image as Image
import glob
import pickle as pkl

class Iterator(object):


    def __init__(self, root_path="inpainting", img_path = 'train2014',
                 caps_path='dict_key_imgID_value_caps_train_and_valid.pkl',
                 batch_size=128, nb_sub=None, extract_center=True, load_caption=False):


        self.root_path = root_path
        self.img_path = os.path.join(root_path, img_path)
        self.caps_path = os.path.join(root_path, caps_path)
        self.batch_size = batch_size
        self.batch_idx = 0
        self.imgs = glob.glob(self.img_path + "/*.jpg")
        self.extract_center = extract_center
        self.load_caption = load_caption

        if nb_sub is not None:
            self.imgs = self.imgs[:nb_sub]

        if load_caption:
            with open(self.caps_path) as fd:
                print "Loading the captions..."
                self.caption_dict = pkl.load(fd)
                print "Done"


    def _get_img(self, i):

        img_path = self.imgs[i]
        img = Image.open(img_path)
        img_array = np.array(img)

        cap_id = os.path.basename(img_path)[:-4]

        ### Get input/target from the images
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            input = np.copy(img_array)
            if self.extract_center:
                input[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = 0
            target = img_array[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :]
        else:
            # Ignore the gray images.
            return None

        cap = None
        if self.load_caption:
            cap= self.caption_dict[cap_id]

        return input.astype('float32')/255., target.astype('float32')/255., cap

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, key):


        if isinstance(key, slice):
            # Get the start, stop, and step from the slice

            res = [self[ii] for ii in xrange(*key.indices(len(self)))]
            xs, ys, caps = zip(*[x for x in res if x is not None])
            return np.array(xs), np.array(ys), caps

        elif isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError, "The index (%d) is out of range." % key
            return self._get_img(key)  # Get the data from elsewhere
        else:
            raise TypeError, "Invalid argument type."


    def __iter__(self):

        for batch_idx in range(int(len(self)/self.batch_size)):

            if (batch_idx+1)*self.batch_size < len(self):
                yield self[batch_idx*self.batch_size: (batch_idx+1)*self.batch_size]
            else:
                yield self[batch_idx * self.batch_size:]

class PreprocessIterator(Iterator):

    def __init__(self, root_path="inpainting", img_path = 'preprocess',
                 caps_path='dict_key_imgID_value_caps_train_and_valid.pkl',
                 batch_size=128, nb_sub=None, extract_center=True, load_caption=False):

        self.root_path = root_path
        self.img_path = os.path.join(root_path, img_path)
        self.caps_path = os.path.join(root_path, caps_path)
        self.batch_size = batch_size
        self.batch_idx = 0
        self.imgs_x = sorted(glob.glob(self.img_path + "/*_x*.npy"))
        self.imgs_y = sorted(glob.glob(self.img_path + "/*_y*.npy"))
        self.nb_sub = nb_sub
        self.extract_center = extract_center
        self.load_caption = load_caption

        if load_caption:
            with open(self.caps_path) as fd:
                print "Loading the captions..."
                self.caption_dict = pkl.load(fd)
                print "Done"

        self._get_lookup_table()
        self.cache = {'file':self._get_file(0),
                      'xs': self._load(self._get_file(0)),
                      'ys': self._load(self._get_file(0, 'y'))}

    def _get_file(self, i, sub='x'):
        return self.img_path + "/data_train_{}_{}.npy".format(sub, i)

    def _load(self, file_name):
        return np.load(open(file_name))

    def __len__(self):
        return len(self.lookup)

    def _get_lookup_table(self):

        print "processing the lookup table..."
        self.lookup = {}

        sum_idx = 0
        for ff in self.imgs_x:

            print "Doing {}...".format(ff)
            img = np.load(open(ff))
            nb = img.shape[0]
            for i in range(nb):
                self.lookup[sum_idx + i] = (i, ff)
            sum_idx += nb

    def _get_img(self, i):

        img_no, img_file = self.lookup[i]
        if img_file != self.cache['file']:
            print "{} isn't in cache. Loading now".format(img_file)
            del self.cache['xs']
            del self.cache['ys']
            self.cache['xs'] = self._load(img_file)
            self.cache['ys'] = self._load(img_file.replace('x', 'y'))
            self.cache['file'] = img_file

        input = self.cache['xs'][img_no]
        target = self.cache['ys'][img_no]

        cap = None
        if self.load_caption:
            cap_id = os.path.basename(img_path)[:-4]
            cap= self.caption_dict[cap_id]

        return input, target, cap