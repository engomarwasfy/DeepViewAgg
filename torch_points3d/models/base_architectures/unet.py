import copy
import importlib
from abc import ABC
from torch import nn
from torch_points3d.core.common_modules.base_modules import Identity
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.core.multimodal.data import MMData, MODALITY_NAMES
from torch_points3d.models.base_model import BaseModel
from torch_points3d.modules.multimodal.modules import MultimodalBlockDown, \
    UnimodalBranch
from torch_points3d.utils.config import is_list, get_from_kwargs, \
    fetch_arguments_from_list, flatten_compact_options, fetch_modalities
from omegaconf.listconfig import ListConfig
import logging

log = logging.getLogger(__name__)

SPECIAL_NAMES = ["radius", "max_num_neighbors", "block_names"]


class BaseFactory:
    def __init__(self, module_name_down, module_name_up, modules_lib):
        self.module_name_down = module_name_down
        self.module_name_up = module_name_up
        self.modules_lib = modules_lib

    def get_module(self, flow):
        if flow.upper() == "UP":
            return getattr(self.modules_lib, self.module_name_up, None)
        else:
            return getattr(self.modules_lib, self.module_name_down, None)


def get_factory(model_name, modules_lib) -> BaseFactory:
    factory_module_cls = getattr(modules_lib, "{}Factory".format(model_name),
                                 None)
    if factory_module_cls is None:
        factory_module_cls = BaseFactory
    return factory_module_cls


class ModalityFactory:
    """Factory for building modality-specific convolutional modules and
    merge modules.

     Modules are expected to be found in:
        modules.multimodal.modalities.<modality>.module_name
     """

    def __init__(self, modality, module_name, atomic_pooling_name,
                 view_pooling_name, fusion_name):
        self.modality = modality

        self.module_name = module_name
        self.modality_lib = importlib.import_module(
            f"torch_points3d.modules.multimodal.modalities.{modality}")

        self.atomic_pooling_name = atomic_pooling_name
        self.view_pooling_name = view_pooling_name
        self.pooling_lib = importlib.import_module(
            f"torch_points3d.modules.multimodal.pooling")

        self.fusion_name = fusion_name
        self.fusion_lib = importlib.import_module(
            f"torch_points3d.modules.multimodal.fusion")

    def get_module(self, flow):
        if flow.upper() == 'ATOMIC':
            # Search for the modality pooling in
            # torch_points3d.modules.multimodal.pooling
            return getattr(self.pooling_lib, self.atomic_pooling_name, None)
        elif flow.upper() == 'VIEW':
            # Search for the modality pooling in
            # torch_points3d.modules.multimodal.pooling
            return getattr(self.pooling_lib, self.view_pooling_name, None)
        elif flow.upper() == 'FUSION':
            #  Search for the modality fusion in
            # torch_points3d.modules.multimodal.fusion
            return getattr(self.fusion_lib, self.fusion_name, None)
        elif flow.upper() == 'UNET':
            # Search for the modality UNet in
            # torch_points3d.modules.multimodal.modalities.{modality}
            return getattr(self.modality_lib, 'UNet', None)
        else:
            # Search for the modality conv in
            # torch_points3d.modules.multimodal.modalities.{modality}
            return getattr(self.modality_lib, self.module_name, None)


# ----------------------------- UNET BASE ---------------------------- #

class UnetBasedModel(BaseModel, ABC):
    """Create a Unet-based generator"""

    def __init__(self, opt, model_type, dataset: BaseDataset, modules_lib):
        """Construct a Unet generator
        Parameters:
            opt - options for the network generation
            model_type - type of the model to be generated
            num_class - output of the network
            modules_lib - all modules that can be used in the UNet
        We construct the U-Net from the innermost layer to the
        outermost layer. It is a recursive process.

        opt is expected to contains the following keys:
        * down_conv
        * up_conv
        * OPTIONAL: innermost
        """
        opt = copy.deepcopy(opt)
        super(UnetBasedModel, self).__init__(opt)
        self._spatial_ops_dict = {"neighbour_finder": [], "sampler": [],
                                  "upsample_op": []}
        # Detect which options format has been used to define the model
        if type(opt.down_conv) is ListConfig \
                or "down_conv_nn" not in opt.down_conv:
            self._init_from_layer_list_format(opt, model_type, dataset,
                                              modules_lib)
        else:
            self._init_from_compact_format(opt, model_type, dataset,
                                           modules_lib)

    def _init_from_compact_format(self, opt, model_type, dataset, modules_lib):
        """Create a unetbasedmodel from the compact options format -
        where the same convolution is given for each layer, and
        arguments are given in lists.
        """
        num_convs = len(opt.down_conv.down_conv_nn)

        # Factory for creating up and down modules
        factory_module_cls = get_factory(model_type, modules_lib)
        down_conv_cls_name = opt.down_conv.module_name
        up_conv_cls_name = opt.up_conv.module_name
        self._factory_module = factory_module_cls(
            down_conv_cls_name, up_conv_cls_name, modules_lib
        )  # Create the factory object
        # construct unet structure
        has_innermost = hasattr(opt, "innermost") and opt.innermost is not None
        if has_innermost:
            assert len(opt.down_conv.down_conv_nn) + 1 \
                   == len(opt.up_conv.up_conv_nn)

            args_up = fetch_arguments_from_list(opt.up_conv, 0, SPECIAL_NAMES)
            args_up["up_conv_cls"] = self._factory_module.get_module("UP")

            unet_block = UnetSkipConnectionBlock(
                args_up=args_up,
                args_innermost=opt.innermost,
                modules_lib=modules_lib,
                submodule=None,
                innermost=True,
            )  # add the innermost layer
        else:
            unet_block = Identity()

        if num_convs > 1:
            for index in range(num_convs - 1, 0, -1):
                args_up, args_down = self._fetch_arguments_up_and_down(
                    opt, index)
                unet_block = UnetSkipConnectionBlock(
                    args_up=args_up, args_down=args_down, submodule=unet_block)
                self._save_sampling_and_search(unet_block)
        else:
            index = num_convs

        index -= 1
        args_up, args_down = self._fetch_arguments_up_and_down(opt, index)
        self.model = UnetSkipConnectionBlock(
            args_up=args_up, args_down=args_down, submodule=unet_block,
            outermost=True)  # add the outermost layer
        self._save_sampling_and_search(self.model)

    def _init_from_layer_list_format(self, opt, model_type, dataset,
                                     modules_lib):
        """Create a unetbasedmodel from the layer list options format -
        where each layer of the unet is specified separately.
        """

        get_factory(model_type, modules_lib)

        down_conv_layers = (
            opt.down_conv if type(opt.down_conv) is ListConfig
            else flatten_compact_options(opt.down_conv)
        )
        up_conv_layers = opt.up_conv if type(opt.up_conv) is ListConfig \
            else flatten_compact_options(opt.up_conv)
        num_convs = len(down_conv_layers)

        unet_block = []
        has_innermost = hasattr(opt, "innermost") and opt.innermost is not None
        if has_innermost:
            assert len(down_conv_layers) + 1 == len(up_conv_layers)

            up_layer = dict(up_conv_layers[0])
            up_layer["up_conv_cls"] = getattr(modules_lib,
                                              up_layer["module_name"])

            unet_block = UnetSkipConnectionBlock(
                args_up=up_layer,
                args_innermost=opt.innermost,
                modules_lib=modules_lib,
                innermost=True,
            )

        for index in range(num_convs - 1, 0, -1):
            down_layer = dict(down_conv_layers[index])
            up_layer = dict(up_conv_layers[num_convs - index])

            down_layer["down_conv_cls"] = getattr(modules_lib,
                                                  down_layer["module_name"])
            up_layer["up_conv_cls"] = getattr(modules_lib,
                                              up_layer["module_name"])

            unet_block = UnetSkipConnectionBlock(
                args_up=up_layer,
                args_down=down_layer,
                modules_lib=modules_lib,
                submodule=unet_block,
            )

        up_layer = dict(up_conv_layers[-1])
        down_layer = dict(down_conv_layers[0])
        down_layer["down_conv_cls"] = getattr(modules_lib,
                                              down_layer["module_name"])
        up_layer["up_conv_cls"] = getattr(modules_lib,
                                          up_layer["module_name"])
        self.model = UnetSkipConnectionBlock(
            args_up=up_layer, args_down=down_layer, submodule=unet_block,
            outermost=True)

        self._save_sampling_and_search(self.model)

    def _save_sampling_and_search(self, submodule):
        sampler = getattr(submodule.down, "sampler", None)
        if is_list(sampler):
            self._spatial_ops_dict["sampler"] \
                = sampler + self._spatial_ops_dict["sampler"]
        else:
            self._spatial_ops_dict["sampler"] \
                = [sampler] + self._spatial_ops_dict["sampler"]

        neighbour_finder = getattr(submodule.down, "neighbour_finder", None)
        if is_list(neighbour_finder):
            self._spatial_ops_dict["neighbour_finder"] \
                = neighbour_finder + self._spatial_ops_dict["neighbour_finder"]
        else:
            self._spatial_ops_dict["neighbour_finder"] \
                = [neighbour_finder] + self._spatial_ops_dict["neighbour_finder"]

        upsample_op = getattr(submodule.up, "upsample_op", None)
        if upsample_op:
            self._spatial_ops_dict["upsample_op"].append(upsample_op)

    def _fetch_arguments_up_and_down(self, opt, index):
        # Defines down arguments
        args_down = fetch_arguments_from_list(opt.down_conv, index,
                                              SPECIAL_NAMES)
        args_down["index"] = index
        args_down["down_conv_cls"] = self._factory_module.get_module("DOWN")

        # Defines up arguments
        idx = len(getattr(opt.up_conv, "up_conv_nn")) - index - 1
        args_up = fetch_arguments_from_list(opt.up_conv, idx, SPECIAL_NAMES)
        args_up["index"] = index
        args_up["up_conv_cls"] = self._factory_module.get_module("UP")
        return args_up, args_down


class UnetSkipConnectionBlock(nn.Module, ABC):
    """Defines the Unet submodule with skip connection.
    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|

    """

    def __init__(
            self,
            args_up=None,
            args_down=None,
            args_innermost=None,
            modules_lib=None,
            submodule=None,
            outermost=False,
            innermost=False,
    ):
        """Construct a Unet submodule with skip connections.
        Parameters:
            args_up -- arguments for up convs
            args_down -- arguments for down convs
            args_innermost -- arguments for innermost
            submodule (UnetSkipConnectionBlock) -- previously defined
                submodules
            outermost (bool)    -- if this module is the outermost
                module
            innermost (bool)    -- if this module is the innermost
                module
        """
        super(UnetSkipConnectionBlock, self).__init__()

        self.outermost = outermost
        self.innermost = innermost

        if innermost:
            assert outermost == False
            module_name = get_from_kwargs(args_innermost, "module_name")
            inner_module_cls = getattr(modules_lib, module_name)
            self.inner = inner_module_cls(**args_innermost)
            upconv_cls = get_from_kwargs(args_up, "up_conv_cls")
            self.up = upconv_cls(**args_up)
        else:
            downconv_cls = get_from_kwargs(args_down, "down_conv_cls")
            upconv_cls = get_from_kwargs(args_up, "up_conv_cls")
            downconv = downconv_cls(**args_down)
            upconv = upconv_cls(**args_up)

            self.down = downconv
            self.submodule = submodule
            self.up = upconv

    def forward(self, data, *args, **kwargs):
        if self.innermost:
            data_out = self.inner(data, **kwargs)
            data = (data_out, data)
            return self.up(data, **kwargs)
        else:
            data_out = self.down(data, **kwargs)
            data_out2 = self.submodule(data_out, **kwargs)
            data = (data_out2, data)
            return self.up(data, **kwargs)


# ------------------------ UNWRAPPED UNET BASE ----------------------- #

class UnwrappedUnetBasedModel(BaseModel, ABC):
    """Create a Unet unwrapped generator. Supports multimodal encoding.
    """

    def __init__(self, opt, model_type, dataset: BaseDataset, modules_lib):
        """Construct a Unet unwrapped generator. Supports multimodal
        encoding.

        The layers will be appended within lists with the following
        names:
        * down_modules : Contains all the down module - may be
            multimodal
        * inner_modules : Contain one or more inner modules
        * up_modules: Contains all the up module

        Parameters:
            opt - options for the network generation
            model_type - type of the model to be generated
            num_class - output of the network
            modules_lib - all modules that can be used in the UNet

        For a recursive implementation. See UnetBaseModel.

        opt is expected to have the following format:
            down_conv:
                module_name: ...
                down_conv_nn: ...
                *args

                <modality_name>: [OPTIONAL]
                    module_name: ...
                    down_conv_nn: ...
                    *args

                    merge:
                        module_name: ...
                        *args

            innermost: [OPTIONAL]
                module_name: ...
                *args

            up_conv:
                module_name: ...
                up_conv_nn: ...
                *args
        """
        opt = copy.deepcopy(opt)
        super(UnwrappedUnetBasedModel, self).__init__(opt)
        self._spatial_ops_dict = {"neighbour_finder": [], "sampler": [],
                                  "upsample_op": []}

        # Check if one of the supported modalities is present in the
        # config
        self._modalities = fetch_modalities(opt.down_conv, MODALITY_NAMES)

        # Detect which options format has been used to define the model
        if is_list(opt.down_conv) or "down_conv_nn" not in opt.down_conv:
            raise NotImplementedError
        else:
            self._init_from_compact_format(opt, model_type, dataset,
                                           modules_lib)

    def _init_from_compact_format(self, opt, model_type, dataset, modules_lib):
        """Create a unetbasedmodel from the compact options format -
        where the same convolution is given for each layer, and
        arguments are given in lists.
        """
        self.save_sampling_id = opt.down_conv.save_sampling_id

        # Factory for creating up and down modules for the main 3D
        # modality
        factory_module_cls = get_factory(model_type, modules_lib)
        down_conv_cls_name = opt.down_conv.module_name
        up_conv_cls_name = opt.up_conv.module_name if opt.up_conv is not None \
            else None
        self._module_factories = {'main': factory_module_cls(
            down_conv_cls_name, up_conv_cls_name, modules_lib)}

        # Factories for creating modules for additional modalities
        if self.is_multimodal:
            for m in self.modalities:
                mod_opt = getattr(opt.down_conv, m)
                self._module_factories[m] = ModalityFactory(
                    m,
                    mod_opt.down_conv.module_name,
                    mod_opt.atomic_pooling.module_name,
                    mod_opt.view_pooling.module_name,
                    mod_opt.fusion.module_name)

        # Innermost module - 3D conv only
        self.inner_modules = nn.ModuleList()
        has_innermost = hasattr(opt, "innermost") \
                          and opt.innermost is not None
        if has_innermost:
            inners = self._create_inner_modules(opt.innermost, modules_lib)
            for inner in inners:
                self.inner_modules.append(inner)
        else:
            self.inner_modules.append(Identity())

        # Down modules - 3D conv only
        down_modules = []
        for i in range(len(opt.down_conv.down_conv_nn)):
            down_conv_3d = self._build_module(opt.down_conv, i, flow="DOWN")
            self._save_sampling_and_search(down_conv_3d)
            down_modules.append(down_conv_3d)

        # Down modules - modality-specific branches
        if self.is_multimodal:
            assert len(down_modules) % 2 == 0 and len(down_modules) > 0, \
                f"Expected an even number of 3D conv modules but got " \
                f"{len(down_modules)} modules instead."
            n_layers_down = len(down_modules) // 2

            branches = [{m: nn.Identity() for m in self.modalities}
                        for _ in range(n_layers_down)]

            for m in self.modalities:
                # Get the branching indices
                b_idx = getattr(opt.down_conv, m).branching_index
                b_idx = [b_idx] if not is_list(b_idx) else b_idx

                # Check whether the modality module is a UNet
                is_unet = hasattr(getattr(opt.down_conv, m), 'up_conv') \
                          and getattr(opt.down_conv, m).up_conv is not None
                assert not is_unet or len(b_idx) == 1, \
                    f"Cannot build a {m}-specific UNet with multiple " \
                    f"branching indices. Consider removing the 'up_conv' " \
                    f"from the {m} modality or providing a single branching " \
                    f"index."

                # Build the branches
                for i, idx in enumerate(b_idx):
                    if is_unet:
                        unet_cls = self._module_factories[m].get_module('UNET')
                        conv = unet_cls(getattr(opt.down_conv, m))
                    else:
                        conv = self._build_module(
                            getattr(opt.down_conv, m).down_conv, i, modality=m)
                    atomic_pool = self._build_module(
                        getattr(opt.down_conv, m).atomic_pooling, i, modality=m,
                        flow='ATOMIC')
                    view_pool = self._build_module(
                        getattr(opt.down_conv, m).view_pooling, i, modality=m,
                        flow='VIEW')
                    fusion = self._build_module(
                        getattr(opt.down_conv, m).fusion, i, modality=m,
                        flow='FUSION')

                    # Group modules into a UnimodalBranch and update the
                    # branches at the proper branching point
                    branches[idx][m] = UnimodalBranch(conv, atomic_pool,
                                                      view_pool, fusion)

            # Update the down_modules list
            down_modules = [
                MultimodalBlockDown(conv_1, conv_2, **modal_conv)
                for conv_1, conv_2, modal_conv
                in zip(down_modules[::2], down_modules[1::2], branches)]

        # Down modules - combined
        self.down_modules = nn.ModuleList(down_modules)

        # Up modules - 3D conv only
        self.up_modules = nn.ModuleList()
        if up_conv_cls_name:
            for i in range(len(opt.up_conv.up_conv_nn)):
                up_module = self._build_module(opt.up_conv, i, flow="UP")
                self._save_upsample(up_module)
                self.up_modules.append(up_module)

        self.metric_loss_module, self.miner_module = \
            BaseModel.get_metric_loss_and_miner(
                getattr(opt, "metric_loss", None), getattr(opt, "miner", None))

    def _save_sampling_and_search(self, down_conv):
        sampler = getattr(down_conv, "sampler", None)
        if is_list(sampler):
            self._spatial_ops_dict["sampler"] += sampler
        else:
            self._spatial_ops_dict["sampler"].append(sampler)

        neighbour_finder = getattr(down_conv, "neighbour_finder", None)
        if is_list(neighbour_finder):
            self._spatial_ops_dict["neighbour_finder"] += neighbour_finder
        else:
            self._spatial_ops_dict["neighbour_finder"].append(neighbour_finder)

    def _save_upsample(self, up_conv):
        upsample_op = getattr(up_conv, "upsample_op", None)
        if upsample_op:
            self._spatial_ops_dict["upsample_op"].append(upsample_op)

    def _collect_sampling_ids(self, list_data):

        def extract_matching_key(keys, start_token):
            for key in keys:
                if key.startswith(start_token):
                    return key
            return None

        d = {}
        if self.save_sampling_id:
            for idx, data in enumerate(list_data):
                if isinstance(data, MMData):
                    data = data.data
                key = extract_matching_key(data.keys, "sampling_id")
                if key:
                    d[key] = getattr(data, key)
        return d

    def _create_inner_modules(self, args_innermost, modules_lib):
        inners = []
        if is_list(args_innermost):
            for inner_opt in args_innermost:
                module_name = get_from_kwargs(inner_opt, "module_name")
                inner_module_cls = getattr(modules_lib, module_name)
                inners.append(inner_module_cls(**inner_opt))

        else:
            module_name = get_from_kwargs(args_innermost, "module_name")
            inner_module_cls = getattr(modules_lib, module_name)
            inners.append(inner_module_cls(**args_innermost))

        return inners

    def _build_module(self, conv_opt, index, flow='DOWN', modality='main'):
        """Builds a convolution (up or down) or a merge block in the
        case of multimodal models.

        Arguments:
            conv_opt - model config subset describing the convolutional
                block
            index - layer index in sequential order (as they come in
                the config)
            flow - "UP", "DOWN", "ATOMIC, "VIEW" or "FUSION"
            modality - string among supported modalities
        """
        args = fetch_arguments_from_list(conv_opt, index, SPECIAL_NAMES)
        args["index"] = index
        module = self._module_factories[modality].get_module(flow)
        return module(**args)

    def forward(self, data, precomputed_down=None, precomputed_up=None,
                **kwargs):
        """This method does a forward on the Unet assuming symmetrical
        skip connections

        Parameters
        ----------
        data: torch.geometric.Data
            Data object that contains all info required by the modules
        precomputed_down: torch.geometric.Data
            Precomputed data that will be passed to the down convs
        precomputed_up: torch.geometric.Data
            Precomputed data that will be passed to the up convs
        """
        # TODO : expand to handle multimodal data or let child classes handle it ?
        if self.is_multimodal:
            raise NotImplementedError

        stack_down = []
        for i in range(len(self.down_modules) - 1):
            data = self.down_modules[i](data, precomputed=precomputed_down)
            stack_down.append(data)
        data = self.down_modules[-1](data, precomputed=precomputed_down)

        if not isinstance(self.inner_modules[0], Identity):
            stack_down.append(data)
            data = self.inner_modules[0](data)

        sampling_ids = self._collect_sampling_ids(stack_down)

        for i in range(len(self.up_modules)):
            data = self.up_modules[i]((data, stack_down.pop()),
                                      precomputed=precomputed_up)

        for key, value in sampling_ids.items():
            setattr(data, key, value)
        return data
