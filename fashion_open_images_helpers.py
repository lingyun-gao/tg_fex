# We refer to 'lyst_external' data source as 'fashion' for brevity

import lyst_external_helpers as lyst_external
import open_images_helpers as open_images


class FashionOpenImages:
    num_classes = lyst_external.LystExternal.num_classes + \
        open_images.OpenImages.num_classes

    modules_specs = lyst_external.LystExternal.modules_specs + \
        open_images.OpenImages.modules_specs

    label2name = lyst_external.LystExternal.num2name.copy()
    label2name.update(open_images.OpenImages.wnid2name)

    idx2name = lyst_external.LystExternal.idx2name.copy()  # per module
    idx2name[open_images.open_images_mod_name] = open_images.OpenImages.idx2name
