import os
import time
import pickle
import random
import socket
import hashlib
import traceback
import multiprocessing as mp
from os.path import join, exists
import numpy as np
from skimage.transform import resize
from PIL import Image, ImageOps, ImageEnhance

# for making index
if socket.gethostname() == 'billy':
    DATA_ROOT = '/home/data/training_data'  # on billy
else:
    DATA_ROOT = '/mnt'  # on spot-billy


# for jitter
ASPECT_LO = 3. / 4
ASPECT_HI = 4. / 3
CROP_LO = 3. / 4
CROP_HI = 1.
ROTATE_PROB = 0.3


COLORS_LABELS = set([
    92, 153, 154, 155, 156, 157, 158, 161, 162, 164, 165, 166, 167, 168, 169,
    170, 172, 173, 174, 178, 180, 181, 183, 190, 192, 194, 195, 196, 198, 203,
    204, 205, 206, 208, 209, 215, 218, 219, 221, 225, 227, 228, 229, 232, 233,
    234, 235, 243, 245, 246, 248, 285, 375, 409, 712
])
PATTERNS_LABELS = set([
    1, 6, 8, 18, 31, 38, 47, 62, 77, 87, 91, 97, 98, 99, 101, 110, 112, 126,
    132, 134, 163, 263, 268, 269, 277, 278, 283, 284, 302, 317, 318, 335, 338,
    340, 341, 344, 352, 354, 355, 357, 367, 369, 371, 380, 382, 390, 401, 403,
    406, 413, 423, 438, 439, 447, 450, 476, 492, 494, 502, 509, 510, 522, 525,
    526, 536, 548, 549, 551, 558, 570, 571, 573, 586, 595, 596, 597, 599, 601,
    604, 615, 621, 623, 645, 646, 655, 673, 684, 690, 693, 694, 719, 721, 738,
    739, 741, 752, 753, 771, 773, 804, 837, 846, 847, 886, 887
])
DETAILS_LABELS = set([
    5, 7, 30, 37, 44, 46, 60, 61, 67, 68, 70, 76, 84, 108, 113, 117, 128, 142,
    251, 253, 255, 258, 261, 282, 299, 303, 304, 307, 314, 316, 321, 322, 326,
    329, 331, 332, 336, 337, 353, 360, 362, 379, 383, 384, 385, 394, 395, 414,
    415, 426, 427, 430, 431, 436, 473, 482, 493, 497, 507, 511, 524, 528, 530,
    547, 559, 561, 569, 572, 575, 578, 580, 581, 590, 592, 600, 602, 610, 624,
    628, 629, 636, 639, 643, 648, 649, 658, 675, 691, 695, 696, 701, 707, 708,
    709, 710, 713, 714, 725, 731, 742, 750, 755, 756, 759, 763, 780, 784, 801,
    813, 828, 839, 844, 845, 856, 857, 862, 864, 888, 889, 890, 891, 892, 893
])
SHAPES_LABELS = set([
    2, 3, 4, 9, 10, 12, 13, 16, 17, 19, 20, 21, 22, 23, 25, 26, 27, 29, 32, 33,
    34, 35, 36, 39, 40, 41, 42, 43, 45, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    58, 59, 63, 64, 65, 66, 69, 72, 73, 74, 75, 78, 79, 80, 81, 82, 83, 85, 86,
    88, 89, 90, 93, 94, 100, 102, 104, 106, 107, 109, 111, 114, 115, 116, 118,
    119, 121, 123, 124, 125, 127, 131, 133, 135, 136, 137, 138, 139, 140, 141,
    144, 145, 147, 148, 250, 252, 256, 257, 259, 260, 265, 266, 270, 271, 272,
    273, 274, 279, 280, 286, 287, 288, 289, 290, 292, 293, 294, 295, 296, 297,
    300, 301, 306, 308, 309, 310, 311, 312, 313, 315, 319, 320, 323, 324, 325,
    328, 339, 342, 343, 346, 347, 348, 349, 350, 351, 358, 361, 363, 364, 365,
    366, 368, 370, 372, 373, 374, 378, 386, 388, 389, 391, 393, 397, 398, 399,
    400, 404, 410, 420, 422, 428, 429, 432, 433, 434, 437, 440, 443, 444, 451,
    452, 453, 454, 456, 457, 458, 461, 463, 464, 466, 468, 469, 470, 471, 474,
    475, 477, 478, 481, 484, 485, 488, 496, 498, 499, 501, 503, 508, 512, 516,
    518, 519, 521, 523, 527, 529, 537, 538, 539, 540, 544, 545, 550, 553, 554,
    555, 556, 560, 562, 563, 565, 566, 568, 574, 577, 579, 583, 584, 585, 588,
    589, 591, 593, 598, 603, 605, 606, 608, 609, 614, 616, 617, 618, 619, 620,
    622, 625, 626, 627, 630, 631, 637, 640, 642, 650, 652, 653, 654, 656, 657,
    659, 661, 663, 664, 665, 666, 667, 669, 671, 672, 674, 676, 677, 678, 679,
    680, 681, 683, 686, 687, 688, 689, 698, 699, 702, 703, 705, 711, 716, 718,
    720, 722, 723, 728, 729, 730, 732, 733, 734, 735, 736, 740, 743, 746, 747,
    751, 754, 760, 761, 762, 764, 765, 766, 767, 768, 769, 772, 774, 775, 777,
    778, 779, 781, 782, 783, 785, 786, 787, 789, 792, 793, 796, 798, 799, 800,
    802, 803, 809, 810, 811, 814, 815, 816, 817, 818, 819, 821, 822, 823, 824,
    826, 827, 832, 833, 835, 838, 840, 841, 842, 848, 849, 851, 855, 859, 860,
    861, 863, 865, 866, 867, 868, 869, 870, 871, 873, 874, 875, 876, 877, 878,
    879, 880, 881, 882, 883, 884, 885, 894, 895
])
MODULES = [
    ('color', COLORS_LABELS),
    ('pattern', PATTERNS_LABELS),
    ('detail', DETAILS_LABELS),
    ('shape', SHAPES_LABELS)
]
MODULES_SPECS = [(mod_name, len(mod_set)) for mod_name, mod_set in MODULES]


class Transformer():
    def __init__(self):
        mean = np.array([104.00698793, 116.66876762, 122.67891434])
        self.MEAN_VALUE = mean[:, np.newaxis, np.newaxis]

    def transform_image(self, img, img_size=224):
        # Replicates caffe.io.Transformer
        # Takes as input [0, 255] image

        # Resize image
        img = resize(img, (img_size, img_size), preserve_range=True)

        # Shuffle axes to c01
        img = np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)

        # Convert to BGR
        img = img[::-1, :, :]

        # Set mean value
        img -= self.MEAN_VALUE

        return img


def get_maps(model_prefix):
    # initialize maps
    num2idx = {}
    idx2num = {}
    idx2name = {}
    for mod_name, _ in MODULES_SPECS:
        num2idx[mod_name] = {}
        idx2num[mod_name] = {}
        idx2name[mod_name] = {}

    canonical_labels_fn = join(model_prefix, 'canonical_labels.txt')
    with open(canonical_labels_fn, 'r') as f_in:
        idx = {}
        for mod_name, _ in MODULES_SPECS:
            idx[mod_name] = 0
        for line in f_in:
            num, name = line.strip().split(' ', 1)
            num = int(num)
            for mod_name, mod_set in MODULES:
                if num in mod_set:
                    num2idx[mod_name][num] = idx[mod_name]
                    idx2num[mod_name][idx[mod_name]] = num
                    idx2name[mod_name][idx[mod_name]] = name
                    idx[mod_name] += 1

    return (num2idx, idx2num, idx2name)


class ReadData():

    def __init__(self, model_prefix):
        self.model_prefix = model_prefix
        self.num2idx, self.idx2num, \
            self.idx2name, = get_maps(model_prefix)
        self.modules = MODULES
        self.modules_specs = MODULES_SPECS
        self.n_modules = len(MODULES_SPECS)
        self.n_labels = sum([mod_size for _, mod_size in MODULES_SPECS])


def avg_image_border(image):
    image_array = np.asarray(image)
    left = image_array[0, :]
    right = image_array[-1, :]
    top = image_array[:, 0]
    bottom = image_array[:, -1]
    average = np.concatenate((left, right, top, bottom)).mean(axis=0)
    return tuple(map(int, average.tolist()))


def open_pad_image(path):
    image = Image.open(path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    w, h = image.size
    d = max(w, h)
    new_im = Image.new('RGB', (d, d), avg_image_border(image))
    new_im.paste(image, ((d - w) / 2, (d - h) / 2))
    return new_im


def get_rotate_crop(w, h):
    assert w == h
    x = int(w / (1 + np.sqrt(2)))
    s = int(x / (2 * np.sqrt(2)))
    return (s, s, 3 * s + x, 3 * s + x)


def rot_sens(idxs):
    pattern_idxs = set(idxs['pattern'])
    if pattern_idxs:
        # horizontal and vertical stripes
        if 401 in pattern_idxs or 804 in pattern_idxs:
            return True
    return False


def color_sens(idxs):
    return bool(idxs['color'])


def generate_jitter(payload):
    path, idxs, img_size = payload
    image = open_pad_image(path)
    w, h = image.size

    if (random.random() < ROTATE_PROB) and not rot_sens(idxs):
        # Perform rotation on padded square image
        image = image.rotate(45 * random.randint(0, 8))
        image = image.crop(get_rotate_crop(w, h))

    else:
        # Sample a random aspect ratio (height / width) crop
        aspect_ratio = ASPECT_LO + random.random() * (ASPECT_HI - ASPECT_LO)
        # Sample a random patch size between 0.1x and 1.0x (in terms of width)
        crop_w_lo = min(w, h / aspect_ratio) * CROP_LO
        crop_w_hi = min(w, h / aspect_ratio) * CROP_HI
        crop_w = crop_w_lo + random.random() * (crop_w_hi - crop_w_lo)
        crop_w, crop_h = int(crop_w), int(crop_w * aspect_ratio)
        crop_x = int(random.random() * (w - crop_w))
        crop_y = int(random.random() * (h - crop_h))
        image = image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))

    # Flip horizontally
    if random.random() > 0.5:
        image = ImageOps.mirror(image)

    # # Modify color properties only if label is NOT color
    # if idxs['color']:
    enhancer_clss = [ImageEnhance.Contrast,
                     ImageEnhance.Brightness]
    if not color_sens:
        enhancer_clss += [ImageEnhance.Color]
    random.shuffle(enhancer_clss)
    for enhancer_cls in enhancer_clss:
        enhancer = enhancer_cls(image)
        factor = 0.5 + random.random()
        image = enhancer.enhance(factor)

    image = np.asarray([t.transform_image(np.array(image), img_size)])

    return image, idxs, path


def generate_image(payload, transform_image):
    path, idxs, img_size = payload
    image = open_pad_image(path)

    image = np.asarray([transform_image(np.array(image), img_size)])

    return image, idxs, path


def build_batch(batch):
    n_batch = len(batch)
    images = []
    targets = {}
    target_idxs = {}
    for mod_name, mod_size in MODULES_SPECS:
        targets[mod_name] = np.zeros((n_batch, mod_size))
        target_idxs[mod_name] = []
    paths = []
    for b_idx, (im, idxs, p) in enumerate(batch):
        images.append(im)
        for mod_name, _ in MODULES_SPECS:
            if idxs[mod_name]:
                targets[mod_name][b_idx, idxs[mod_name]] = 1.0 / len(idxs[mod_name])
                target_idxs[mod_name].append(b_idx)
        paths.append(p)

    images = np.concatenate(images)
    for mod_name, _ in MODULES_SPECS:
        idxs[mod_name] = np.array(idxs[mod_name])
    return images, targets, target_idxs, paths


def read_train_file_loop(model_prefix, infinite=True):
    training_data_fn = join(model_prefix, 'lyst_external_training_data_multilabeled.txt')
    while True:
        for line in open(training_data_fn, 'r'):
            yield line
        if not infinite:
            raise StopIteration


class ReadBatch():

    def __init__(self, data, batch_size, img_size):
        self.data = data
        self.batch_size = batch_size
        self.img_size = img_size
        self.test_path_hash = set([])
        self.test_dir = join(self.data.model_prefix, 'test_data')
        if not exists(self.test_dir):
            os.mkdir(self.test_dir)

        self.t = Transformer()

    def generate_test_set(self, n_batches):
        if all([exists(join(self.test_dir, '%d.pkl' % i)) for i in xrange(n_batches + 1)]):
            # the 0th pkl is the set of url hashes
            with open(join(self.test_dir, '0.pkl'), 'r') as f:
                self.test_path_hash = pickle.load(f)
            print('Test data already exists in %s, loaded seen set' % self.test_dir)
            return

        # otherwise generate test set
        test_path_hash = set([])
        test_set = self.get_batch(jitter=False, forever=n_batches)
        for idx, test_batch in enumerate(test_set, 1):
            fn = join(self.test_dir, '%d.pkl' % idx)  # batch pickles
            with open(fn, 'w') as f:
                pickle.dump(test_batch, f)
            _, _, _, ps = test_batch
            for p in ps:
                test_path_hash.add(hashlib.sha224(p).hexdigest())
        fn = join(self.test_dir, '0.pkl')  # seen hashes set
        with open(fn, 'w') as f:
            pickle.dump(test_path_hash, f)
        self.test_path_hash = test_path_hash
        print('Saved test data in %s' % self.test_dir)
        return

    def get_test_data(self, n_batches):
        for idx in xrange(1, n_batches + 1):
            with open(join(self.test_dir, '%d.pkl' % idx), 'r') as f:
                test_batch = pickle.load(f)
                yield test_batch

    def get_batch(self, jitter=True, forever=True):

        max_yields = 9e9 if forever and isinstance(forever, bool) else forever

        f = generate_jitter if jitter else generate_image

        pool = mp.Pool(mp.cpu_count() + 2)
        result = None
        batch = []
        n_yields = 0
        n_in_batch = 0
        for line in read_train_file_loop(self.data.model_prefix):
            if random.random() > 0.6:
                continue  # skip random lines

            image_path, nums = line.strip().split()
            image_path = join(DATA_ROOT, image_path)

            image_path_hash = hashlib.sha224(image_path).hexdigest()
            if image_path_hash in self.test_path_hash:
                continue  # skip path in test data

            nums = map(int, nums.split(','))
            idxs = {}
            for mod_name, _ in MODULES_SPECS:
                idxs[mod_name] = []
            for num in nums:
                for mod_name, mod_set in MODULES:
                    if num in mod_set:
                        idxs[mod_name].append(self.data.num2idx[mod_name][num])
                        break

            batch.append((image_path,
                          idxs,
                          self.img_size))

            n_in_batch += 1

            if n_in_batch != self.batch_size:
                continue

            next_result = pool.map_async(f, (batch, self.t.transform_image))
            if result:
                t0 = time.time()
                r = None
                try:
                    r = build_batch(result.get())
                except:
                    print('Encountered error in data.get_batch()')
                    traceback.print_exc()

                t1 = time.time() - t0
                if t1 > 2:  # greater than 2s
                    print('Warning! Effective time waiting: %.3fs!' % t1)
                if r:
                    n_yields += 1
                    yield r

            result = next_result
            batch = []
            n_in_batch = 0

            if n_yields >= max_yields:
                pool.close()
                break

        if result:
            r = build_batch(result.get())
            if r:
                yield r
