import torch
import json
from torch_points3d.datasets.multimodal.base_dataset import BaseDatasetMM
from torch_points3d.datasets.segmentation.s3dis import *
from torch_geometric.data import Data
from torch_points3d.datasets.multimodal.data import MMData
from torch_points3d.datasets.multimodal.image import ImageData
from torch_points3d.core.data_transform.multimodal.image import PointImagePixelMappingFromId


################################################################################
#                                 S3DIS Utils                                  #
################################################################################

# Images that are located outside of the point clouds and for which we therefore
# cannot recover a trustworthy projection with occlusions
S3DIS_OUTSIDE_IMAGES = [
    # Area 1
    "camera_f62548d255d24e6983b698e34b343b60_hallway_8_frame_equirectangular_domain",
    "camera_78d716cac81c4b0d85a90927b159b77e_hallway_4_frame_equirectangular_domain",
    "camera_95fab66e5ca643cc97aaab4647f145e3_hallway_5_frame_equirectangular_domain",
    "camera_684b940fafd64fd98aef16157c8a96e2_hallway_5_frame_equirectangular_domain",
    "camera_24f42d6efff54b09a34897f69fa11064_hallway_5_frame_equirectangular_domain",
    "camera_e0c041d3b2a94769b1dc86935f983f34_WC_1_frame_equirectangular_domain",
    "camera_1edba7eece574027bab1fa5459fd8cd4_WC_1_frame_equirectangular_domain",

    # Area 2
    "camera_b1d0a684de5d4412bae05b5da4bd6058_conferenceRoom_1_frame_equirectangular_domain",
    "camera_e0acda5a8a544aa8bf575ff0b1cc4557_conferenceRoom_1_frame_equirectangular_domain",
    "camera_76f70ab0399b4062a8a174dac4a5b5d4_hallway_5_frame_equirectangular_domain",
    "camera_087a5c7a06ac47db91f0b3239b9a568c_hallway_5_frame_equirectangular_domain",
    "camera_c87ef2a2ea404851a1ed515e39c19ebc_hallway_12_frame_equirectangular_domain",

    # Area 3
    "camera_1672a6a767af4676a441d2872752d6b5_office_10_frame_equirectangular_domain",
    "camera_711a8a2f5d1c477da742bddfe3b6c15a_office_10_frame_equirectangular_domain",
    "camera_fed3d2b0206b428d836acd7d7a44f85b_office_9_frame_equirectangular_domain",
    "camera_83f59b29737047b9a139cebb8612803d_office_9_frame_equirectangular_domain",
    "camera_274fa02e9d7748589a4a3171fdc148cc_office_10_frame_equirectangular_domain",
    "camera_ede7064adbbe490284373cf8c0cf8bae_lounge_2_frame_equirectangular_domain",
    "camera_d911682267cf458a87bdfe2fbd491c46_lounge_2_frame_equirectangular_domain",
    "camera_1c029f7dc23548cab4ac62429f96eb76_lounge_2_frame_equirectangular_domain",

    # Area 4
    "camera_21d093553b30417e80f382f09ff9173c_hallway_1_frame_equirectangular_domain",
    "camera_b9eca4fa258e4160823cf6b1da447c8f_hallway_1_frame_equirectangular_domain",
    "camera_887c83e5e56d4b4db6867ea493615ada_lobby_1_frame_equirectangular_domain",
    "camera_161eb799efd24b548a8760ae98a16736_lobby_1_frame_equirectangular_domain",
    "camera_3f70cae87b464ef9a9ad9a1b6118e8c1_lobby_1_frame_equirectangular_domain",
    "camera_927a307dc7f5439faae8c42a35aa6e4c_lobby_1_frame_equirectangular_domain",
    "camera_39e8345836e64d8d9d3bacde37f5ee12_hallway_2_frame_equirectangular_domain",
    "camera_0e47c7ac0bfe4720bfff75f1a24cfb56_office_19_frame_equirectangular_domain",

    # Area 6
    "camera_7edd2f07f6be4b25bcc4b3bf330146ff_hallway_6_frame_equirectangular_domain",
    "camera_af9f94170d54489d97d974a9cca06856_hallway_6_frame_equirectangular_domain",
    "camera_247d69e9f0e64242a9330d70eca2ab0c_hallway_6_frame_equirectangular_domain",
    "camera_932e85d85fd74cbb8c9e9e818e250a98_hallway_6_frame_equirectangular_domain",
    "camera_29d9a627823449bb8df926f6ee29d946_hallway_6_frame_equirectangular_domain",
    "camera_d1763aa9c28546efa570296046d7be26_hallway_6_frame_equirectangular_domain",
    "camera_051e24918e884291aebf022719ad572a_office_23_frame_equirectangular_domain",
    "camera_a642c69e7dec4c98aacc3595f1259b03_hallway_2_frame_equirectangular_domain",
    "camera_cff198550a97439eae623dd55452c4c0_hallway_2_frame_equirectangular_domain",
    "camera_5917b1f08a0143b0994d8b3946d02343_hallway_2_frame_equirectangular_domain",
    "camera_629e301c74124597bd41c2ebe4b791b7_hallway_2_frame_equirectangular_domain",
    "camera_8c334635cf8147549ddce70664d70a12_office_23_frame_equirectangular_domain",
    "camera_699cc3c1b5084f828605bcfc8ed8264d_office_23_frame_equirectangular_domain",
    "camera_76b92c8b42e844759a4561f3e67e1007_office_23_frame_equirectangular_domain",
    "camera_d3e9dda66f31417c99c2d346b1797ebd_office_23_frame_equirectangular_domain",
    "camera_49e8cb6e01a448809a7eda23eeb9d4e2_hallway_2_frame_equirectangular_domain",
    "camera_87ea116ead10458e85a923d54be70fcb_hallway_2_frame_equirectangular_domain",
]

# -------------------------------------------------------------------------------

def read_s3dis_pose(json_file):
    # Area 5b poses need a special treatment
    # Need to see the file comes from Area i in the provided filepath
    area_5b = 'area_5b' in json_file.lower()
    area_5a = 'area_5a' in json_file.lower()

    # Loading the Stanford pose json file
    with open(json_file) as f:
        pose_data = json.load(f)

    # XYZ camera position
    xyz = np.array(pose_data['camera_location'])

    # Omega, Phi, Kappa camera pose
    # We define a different pose coordinate system 
    omega, phi, kappa = [np.double(i) for i in pose_data['final_camera_rotation']]
    #     opk = np.array([omega - (np.pi / 2), -phi, -kappa])
    opk = np.array([omega - (np.pi / 2), -phi, -kappa - (np.pi / 2)])

    # Area 5b poses require some rotation and offset corrections
    if area_5b:
        M = np.array([[0, 1, 0],
                      [-1, 0, 0],
                      [0, 0, 1]])
        xyz = M.dot(xyz) + np.array([-4.10, 6.25, 0.0])
        opk = opk + np.array([0, 0, np.pi / 2])

    return xyz, opk


# -------------------------------------------------------------------------------

def s3dis_image_pose_pairs(image_dir, pose_dir, image_suffix='_rgb.png', pose_suffix='_pose.json',
                           skip_names=None, verbose=False):
    """
    Search for all image-pose correspondences in the directories.
    Return the list of image-pose pairs. Orphans are ignored.
    """
    # Search for images and poses
    image_names = sorted([
        osp.basename(x).replace(image_suffix, '')
        for x in glob.glob(osp.join(image_dir, '*' + image_suffix))
    ])
    pose_names = sorted([
        osp.basename(x).replace(pose_suffix, '')
        for x in glob.glob(osp.join(pose_dir, '*' + pose_suffix))
    ])
    
    # Remove images specified by skip_names
    skip_names = skip_names if skip_names is not None else []
    image_names = [x for x in image_names if x not in skip_names]
    pose_names = [x for x in pose_names if x not in skip_names]

    # Print orphans
    if not image_names == pose_names:
        image_orphan = [
            osp.join(image_dir, x + image_suffix)
            for x in set(image_names) - set(pose_names)
        ]
        pose_orphan = [
            osp.join(pose_dir, x + pose_suffix)
            for x in set(pose_names) - set(image_names)
        ]
        print("Could not recover all image-pose correspondences.")
        print(f"  Orphan images : {len(image_orphan)}/{len(image_names)}")
        if verbose:
            for x in image_orphan:
                print(4 * ' ' + '/'.join(x.split('/')[-4:]))
        print(f"  Orphan poses  : {len(pose_orphan)}/{len(pose_names)}")
        if verbose:
            for x in pose_orphan:
                print(4 * ' ' + '/'.join(x.split('/')[-4:]))

    # Only return the recovered pairs
    correspondences = sorted(list(set(image_names).intersection(set(pose_names))))
    pairs = [(
        osp.join(image_dir, x + image_suffix),
        osp.join(pose_dir, x + pose_suffix)
    )
        for x in correspondences]
    return pairs


# -------------------------------------------------------------------------------

def s3dis_image_area(path):
    """S3DIS-specific. Recover the area from the image path."""
    return path.split('/')[-4]


# -------------------------------------------------------------------------------

def s3dis_image_room(path):
    """S3DIS-specific. Recover the room from the image path."""
    return '_'.join(os.path.basename(path).split('_')[2:4])


# -------------------------------------------------------------------------------

def s3dis_image_name(path):
    """S3DIS-specific. Recover the name from the image path."""
    return os.path.basename(path).split('_')[1]


################################################################################
#                         S3DIS Torch Geometric Dataset                        #
################################################################################

class S3DISOriginalFusedMM(InMemoryDataset):
    """
    Multimodal extension of S3DISOriginalFused from 
    torch_points3d.datasets.segmentation.s3dis to 3D with images.
    """

    form_url = ("https://docs.google.com/forms/d/e/",
                "1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1")
    download_url = "https://drive.google.com/uc?id=0BweDykwS9vIobkVPN0wzRzFwTDg&export=download"
    zip_name = "Stanford3dDataset_v1.2_Version.zip"
    path_file = osp.join(DIR, "s3dis.patch")
    file_name = "Stanford3dDataset_v1.2"
    folders = [f"Area_{i}" for i in range(1, 7)]
    num_classes = S3DIS_NUM_CLASSES

    def __init__(
            self,
            root,
            test_area=6,
            split="train",
            transform=None,
            pre_transform=None,
            pre_collate_transform=None,
            pre_filter=None,
            pre_transform_image=None,
            transform_image=None,
            keep_instance=False,
            verbose=False,
            debug=False,
    ):
        assert test_area in list(range(1, 7))

        self.transform = transform
        self.pre_collate_transform = pre_collate_transform
        self.pre_transform_image = pre_transform_image
        self.transform_image = transform_image
        self.test_area = test_area
        self.keep_instance = keep_instance
        self.verbose = verbose
        self.debug = debug
        self._split = split

        super(S3DISOriginalFusedMM, self).__init__(
            root, transform, pre_transform, pre_filter)

        if split == "train":
            path = self.processed_paths[0]
        elif split == "val":
            path = self.processed_paths[1]
        elif split == "test":
            path = self.processed_paths[2]
        elif split == "trainval":
            path = self.processed_paths[3]
        else:
            raise ValueError(f"Split {split} found, but expected either ",
                             "train, val, trainval or test")

        self._load_data(path)

        if split == "test":
            self.raw_test_data = torch.load(self.raw_areas_paths[test_area - 1])

    @property
    def center_labels(self):
        if hasattr(self.data, "center_label"):
            return self.data.center_label
        else:
            return None

    @property
    def raw_file_names(self):
        return self.folders

    @property
    def image_dir(self):
        return osp.join(self.root, 'image')

    @property
    def pre_processed_path(self):
        pre_processed_file_names = "preprocessed.pt"
        return osp.join(self.processed_dir, pre_processed_file_names)

    @property
    def raw_areas_paths(self):
        return [osp.join(self.processed_dir, "raw_area_%i.pt" % i) for i in range(6)]

    @property
    def processed_file_names(self):
        test_area = self.test_area
        return (
                [f"{s}_{test_area}.pt"
                 for s in ["train", "val", "test", "trainval"]]
                + self.raw_areas_paths
                + [self.pre_processed_path]
        )

    @property
    def raw_test_data(self):
        return self._raw_test_data

    @raw_test_data.setter
    def raw_test_data(self, value):
        self._raw_test_data = value

    def download(self):
        raw_folders = os.listdir(self.raw_dir)
        if len(raw_folders) == 0:
            if not osp.exists(osp.join(self.root, self.zip_name)):
                ########################################################################
                # 
                """
                Here download and unzip images from CUSTOM REPO ?

                Image directory strucutre is assumed to be:
                root
                |___images
                    |___area_{1, 2, 3, 4, 5a, 5b, 6}
                        |___pano
                            |___rgb
                                |___original_image_name.png
                            |___pose
                                |___original_image_name.json
                """
                ########################################################################

                log.info("WARNING: You are downloading S3DIS dataset")
                log.info(f"Please, register yourself by filling up the form at {self.form_url}")
                log.info("***")
                log.info("Press any key to continue, or CTRL-C to exit. By ",
                         "continuing, you confirm filling up the form.")
                input("")
                gdown.download(self.download_url, osp.join(self.root, self.zip_name), quiet=False)
            extract_zip(osp.join(self.root, self.zip_name), self.root)
            shutil.rmtree(self.raw_dir)
            os.rename(osp.join(self.root, self.file_name), self.raw_dir)
            shutil.copy(self.path_file, self.raw_dir)
            cmd = f"patch -ruN -p0 -d  {self.raw_dir} < {osp.join(self.raw_dir, 's3dis.patch')}"
            os.system(cmd)
        else:
            intersection = len(set(self.folders).intersection(set(raw_folders)))
            if intersection != 6:
                shutil.rmtree(self.raw_dir)
                os.makedirs(self.raw_dir)
                self.download()

    def process(self):

        # Download, pre_transform and pre_filter raw data
        # ------------------------------------------------
        if not osp.exists(self.pre_processed_path):

            data_files = [
                (f, room_name, osp.join(self.raw_dir, f, room_name))
                for f in self.folders
                for room_name in os.listdir(osp.join(self.raw_dir, f))
                if osp.isdir(osp.join(self.raw_dir, f, room_name))
            ]

            # Gather all data from each area in a List(List(Data))
            data_list = [[] for _ in range(6)]
            if self.debug:
                areas = np.zeros(7)
            for (area, room_name, file_path) in tq(data_files):
                if self.debug:
                    area_idx = int(area.split('_')[-1])
                    if areas[area_idx] == 5:
                        continue
                    else:
                        areas[area_idx] += 1

                area_num = int(area[-1]) - 1
                if self.debug:
                    read_s3dis_format(file_path, room_name, label_out=True, verbose=self.verbose,
                                      debug=self.debug)
                    continue
                else:
                    xyz, rgb, labels, instance_labels, _ = read_s3dis_format(
                        file_path, room_name, label_out=True, verbose=self.verbose,
                        debug=self.debug)

                    # Room orientation correction
                    # 2 rooms need to be rotated by 180° around Z: Area_2/hallway_11 and Area_5/hallway_6
                    if (area_num == 1 and room_name == 'hallway_11') or (area_num == 4 and room_name == 'hallway_6'):
                        xy_center = (xyz[:, 0:2].max(dim=0)[0] + xyz[:, 0:2].min(dim=0)[0]) / 2
                        xyz[:, 0:2] = 2 * xy_center - xyz[:, 0:2]  # equivalent to 180° Z-rotation around the XY-center

                    rgb_norm = rgb.float() / 255.0
                    data = Data(pos=xyz, y=labels, rgb=rgb_norm)

                    if room_name in VALIDATION_ROOMS:
                        data.is_val = torch.ones(data.num_nodes, dtype=bool)
                    else:
                        data.is_val = torch.zeros(data.num_nodes, dtype=bool)

                    if self.keep_instance:
                        data.instance_labels = instance_labels

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    data_list[area_num].append(data)

            raw_areas = cT.PointCloudFusion()(data_list)
            for i, area in enumerate(raw_areas):
                torch.save(area, self.raw_areas_paths[i])

            for area_datas in data_list:
                # Apply pre_transform
                if self.pre_transform is not None:
                    for data in area_datas:
                        data = self.pre_transform(data)

            # Save the data into one big 'preprocessed.pt' file 
            torch.save(data_list, self.pre_processed_path)

        else:
            # Recover the per-area Data list from the 'preprocessed.pt' file
            print('Loading the preprocessed 3D data...')
            data_list = torch.load(self.pre_processed_path)
            print('Done\n')

        if self.debug:
            return

        # Build the data splits and pre_collate them
        # -------------------------------------------
        print('Preparing 3D data splits...')
        # train_data_list = {}
        # val_data_list = {}
        # trainval_data_list = {}

        # for i in range(6):
        #     if i != self.test_area - 1:
        #         train_data_list[i] = []
        #         val_data_list[i] = []
        #         for data in data_list[i]:
        #             is_val = data.is_val
        #             del data.is_val
        #             if is_val:
        #                 val_data_list[i].append(data)
        #             else:
        #                 train_data_list[i].append(data)
        #         trainval_data_list[i] = val_data_list[i] + train_data_list[i]

        # train_data_list = list(train_data_list.values())
        # val_data_list = list(val_data_list.values())
        # trainval_data_list = list(trainval_data_list.values())
        trainval_data_list = data_list[:self.test_area - 1] + data_list[self.test_area:]
        test_data_list = data_list[self.test_area - 1]
        print('Done\n')

        # print('Train data')
        # for i_area, split_area in enumerate(train_data_list):
        #     print(f"Area {i_area + 1 if i_area + 1 < self.test_area else i_area + 2} : {len(split_area)} rooms")
        # print()
        # print('Val data')
        # for i_area, split_area in enumerate(val_data_list):
        #     print(f"Area {i_area + 1 if i_area + 1 < self.test_area else i_area + 2} : {len(split_area)} rooms")
        # print()
        print('Trainval data')
        for i_area, split_area in enumerate(trainval_data_list):
            print(f"Area {i_area + 1 if i_area + 1 < self.test_area else i_area + 2} : {len(split_area)} rooms")
        print()
        print('Test data')
        print(f"Area {self.test_area} : {len(split_area)} rooms")
        print()

        # Run the pre_collate_transform to finalize the data preparation
        # Among other things, the 'origin_id' and 'point_index' are
        # generated here  
        if self.pre_collate_transform:
            print('Running pre-collate on 3D data...')
            log.info("pre_collate_transform ...")
            log.info(self.pre_collate_transform)
            # train_data_list = self.pre_collate_transform(train_data_list)
            # val_data_list = self.pre_collate_transform(val_data_list)
            test_data_list = self.pre_collate_transform(test_data_list)
            trainval_data_list = self.pre_collate_transform(trainval_data_list)
            print('Done\n')

        # Pre_transform_image heavy computation
        # ---------------------------------------
        rooms = [
            (int(f[-1]) - 1, room_name)
            for f in self.folders
            for room_name in os.listdir(osp.join(self.raw_dir, f))
            if osp.isdir(osp.join(self.raw_dir, f, room_name))
        ]
        rooms = [[r[1] for r in rooms if r[0] == i] for i in range(6)]

        # train_image_list = {}
        # val_image_list = {}
        # trainval_image_list = {}
        test_image_list = []
        trainval_image_list = []

        for i in range(6):

            # S3DIS Area 5 images are split into two folders 'area_5a' and
            # 'area_5b' and one of them requires specific treatment for pose
            # reading
            folders = [f"area_{i + 1}"] if i != 4 else ["area_5a", "area_5b"]

            image_info_list = [
                (i_file, *read_s3dis_pose(p_file))
                for folder in folders
                for i_file, p_file in s3dis_image_pose_pairs(
                    osp.join(self.image_dir, folder, 'pano', 'rgb'),
                    osp.join(self.image_dir, folder, 'pano', 'pose'),
                    skip_names=S3DIS_OUTSIDE_IMAGES
                )
            ]
                        
            print(f"Area {i + 1}")
            print(f"    All image info list : {len(image_info_list)} images")
            print(f"    Area : {len(rooms[i])} rooms")

            # Dropping image info for images outside of rooms found during
            # preprocessing
            image_info_list = [x for x in image_info_list if s3dis_image_room(x[0]) in rooms[i]]

            print(f"    Pruned image info list : {len(image_info_list)} images")
            print()

            # Local helper function to combine image info lists into a more
            # convenient ImageData object.
            def info_list_to_image_data(info_list):
                if len(info_list) > 0:
                    path, pos, opk = [list(x) for x in zip(*info_list)]
                    image_data = ImageData(path=np.array(path), pos=torch.Tensor(pos),
                                           opk=torch.Tensor(opk))
                else:
                    image_data = ImageData()
                return image_data

            # Keep all images for the test area
            if i + 1 == self.test_area:
                test_image_list = info_list_to_image_data(image_info_list)
            else:
                trainval_image_list.append(info_list_to_image_data(image_info_list))
            #
            # # Split between train and val room images otherwise
            # else:
            #     train_image_list[i] = []
            #     val_image_list[i] = []
            #
            #     for image_info in image_info_list:
            #         if s3dis_image_room(image_info[0]) in VALIDATION_ROOMS:
            #             val_image_list[i].append(image_info)
            #         else:
            #             train_image_list[i].append(image_info)
            #
            #     print(f"    Images in train : {len(train_image_list[i])}")
            #     print(f"    Images in val : {len(val_image_list[i])}")
            #     print()
            #
            #     trainval_image_list[i] = val_image_list[i] + train_image_list[i]
            #
            #     train_image_list[i] = info_list_to_image_data(train_image_list[i])
            #     val_image_list[i] = info_list_to_image_data(val_image_list[i])
            #     trainval_image_list[i] = info_list_to_image_data(trainval_image_list[i])

        # train_image_list = list(train_image_list.values())
        # val_image_list = list(val_image_list.values())
        # trainval_image_list = list(trainval_image_list.values())

        # Build the mappings for each split, for each area
        print('Running the image pre-transforms...')

        # print('    Trainval image pre-transforms...')
        # #         trainval_data_list, trainval_image_list, trainval_mappings_list = self.pre_transform_image(
        # #             trainval_data_list, trainval_image_list, None)
        # torch.save(self.pre_transform_image(trainval_data_list, trainval_image_list, None),
        #            self.processed_paths[3])
        # del trainval_data_list, trainval_image_list
        # print('Done\n')

        print('    Trainval image pre-transforms...')
        trainval_data_list, trainval_image_list, trainval_mappings_list = self.pre_transform_image(trainval_data_list,
            trainval_image_list, None)
        is_val_list = [data.is_val for data in trainval_data_list]
        for data in trainval_data_list:
            del data.is_val
        torch.save((trainval_data_list, trainval_image_list, trainval_mappings_list), self.processed_paths[3])

        # Local helper to slice a Data object with a torch.Tensor
        def data_slicing(data, selection):
            data_ = data.clone()
            for key, item in data:
                if torch.is_tensor(item) and item.size(0) == data.num_nodes:
                    data_[key] = data_[key][selection]
            return data_

        # Extract and save train data, images mappings from trainval
        torch.save(
            PointImagePixelMappingFromId(key='point_index')(
                [data_slicing(data, ~is_val) for data, is_val in zip(trainval_data_list, is_val_list)],
                trainval_image_list,
                trainval_mappings_list
            ),
            self.processed_paths[0]
        )

        # Extract and save val data, images mappings from trainval
        torch.save(
            PointImagePixelMappingFromId(key='point_index')(
                [data_slicing(data, is_val) for data, is_val in zip(trainval_data_list, is_val_list)],
                trainval_image_list,
                trainval_mappings_list
            ),
            self.processed_paths[1]
        )
        del trainval_data_list, trainval_image_list, trainval_mappings_list

        #         print('    Train image pre-transforms...')
        # #         train_data_list, train_image_list, train_mappings_list = self.pre_transform_image(
        # #             train_data_list, train_image_list, None)
        #         torch.save(self.pre_transform_image(train_data_list, train_image_list, None),
        #             self.processed_paths[0])
        #         del train_data_list, train_image_list

        #         print('    Val image pre-transforms...')
        # #         val_data_list, val_image_list, val_mappings_list = self.pre_transform_image(
        # #             val_data_list, val_image_list, None)
        #         torch.save(self.pre_transform_image(val_data_list, val_image_list, None),
        #             self.processed_paths[1])
        #         del val_data_list, val_image_list

        print('    Test image pre-transforms...')
        #         test_data_list, test_image_list, test_mappings_list = self.pre_transform_image(
        #             test_data_list, test_image_list, None)
        torch.save(self.pre_transform_image(test_data_list, test_image_list, None),
                   self.processed_paths[2])
        del test_data_list, test_image_list
        print('Done\n')

        # Save the Data, ImageData and ForwardStar mappings for each split

    #         self._save_preprocessed_multimodal_data(
    #             (train_data_list, train_image_list, train_mappings_list),
    #             (val_data_list, val_image_list, val_mappings_list),
    #             (test_data_list, test_image_list, test_mappings_list),
    #             (trainval_data_list, trainval_image_list, trainval_mappings_list)
    #         )
    #         self._save_preprocessed_multimodal_data(
    #             (train_data_list, train_image_list, None),
    #             (val_data_list, val_image_list, None),
    #             (test_data_list, test_image_list, test_mappings_list),
    #             (trainval_data_list, trainval_image_list, None)
    #         )

    #     def _save_preprocessed_multimodal_data(self, mm_data_tuple, path):
    #         """
    #         Save the preprocessed data lists. Results are intended to be loaded with
    #         self._load_data().
    #         """
    #         torch.save(mm_data_tuple, path)

    def _load_data(self, path):
        self.data, self.images, self.mappings = torch.load(path)


# -------------------------------------------------------------------------------

class S3DISSphereMM(S3DISOriginalFusedMM):
    """ Small variation of S3DISOriginalFusedMM that allows random sampling of 
    spheres within an Area during training and validation. Spheres have a radius 
    of 2m. If sample_per_epoch is not specified, spheres are taken on a 2m grid.

    http://buildingparser.stanford.edu/dataset.html

    Parameters
    ----------
    root: str
        path to the directory where the data will be saved
    test_area: int
        number between 1 and 6 that denotes the area used for testing
    train: bool
        Is this a train split or not
    pre_collate_transform:
        Transforms to be applied before the data is assembled into samples 
        (apply fusing here for example)
    keep_instance: bool
        set to True if you wish to keep instance data
    sample_per_epoch
        Number of spheres that are randomly sampled at each epoch (-1 for fixed 
        grid)
    radius
        radius of each sphere
    pre_transform
    transform
    pre_filter
    """

    def __init__(self, root, sample_per_epoch=100, radius=2, *args, **kwargs):
        self._sample_per_epoch = sample_per_epoch
        self._radius = radius
        self._grid_sphere_sampling = cT.GridSampling3D(size=radius / 10.0)
        super().__init__(root, *args, **kwargs)

    def __len__(self):
        if self._sample_per_epoch > 0:
            return self._sample_per_epoch
        else:
            return len(self._test_spheres)

    def __getitem__(self, idx):
        """
        Indexing mechanism for the Dataset.
        
        Overwrites the torch_geometric.InMemoryDataset.__getitem__() used for 
        indexing Dataset. Extends its mechanisms to multimodal data.

        Get a 3D points Data sphere sample with image mapping attributes, along
        with the list ok 
        Only supports indexing with int.
        """
        assert isinstance(idx, int), (f"Indexing with {type(idx)} is not supported, only ",
                                      f"{type(int)} are accepted.")

        # Get the 3D point sample and apply transforms
        i_area, data = self.get(self.indices()[idx])
        data = data if self.transform is None else self.transform(data)

        # Get the corresponding images and mappings
        data, images, mappings = self.transform_image(data, self._images[i_area],
                                                      self._mappings[i_area])

        return MMData(data, images, mappings)

    def get(self, idx):
        """
        Get a 3D points Data sample. Does not return multimodal
        attributes.

        Overwrites the torch_geometric.InMemoryDataset.get(), which is called
        from inside the torch_geometric.InMemoryDataset.__getitem__() used for 
        indexing datasets.
        """
        if self._sample_per_epoch > 0:
            # For train datasets
            # Return a random spherical sample and the associated area id
            return self._get_random()
        else:
            # For test and val datasets
            # Return the precomputed sphere at idx and the associated area id
            test_sphere = self._test_spheres[idx].clone()
            i_area = test_sphere.area_id
            delattr(test_sphere, 'area_id')
            return i_area, test_sphere

    def process(self):
        # We have to include this method, otherwise the parent class skips
        # processing.
        super().process()

    def download(self):
        # We have to include this method, otherwise the parent class skips
        # download.
        super().download()

    def _get_random(self):
        """
        S3DISSphereMM has predefined sphere centers accross all areas in the 
        split. The _get_random method randomly picks a center and recovers the 
        sphere-neighborhood for the appropriate S3DISSphereMM._datas[i_area].

        Called if S3DISSphereMM is NOT test set. 
        """
        # Random spheres biased towards getting more low frequency classes
        chosen_label = np.random.choice(self._labels, p=self._label_counts)
        valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
        centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
        centre = valid_centres[centre_idx]

        #         centre_idx = int(random.random() * (self._centres_for_sampling.shape[0] - 1))
        # centre_idx = 27026
        # 21155 # image 0 and 4
        # 5991  # image 1, 4 and 5
        # 18013  # image 4
        # 22478  # image 0 and 1
        # 34867  # image 0 and 5
        # 15768  # image 0 and 5
        # 34268  # image 5
        # 21512  # image 0 and 1  # VERY interesting case
        # 25954  # image 5
        # 17797  # image 1  # chair closeby caused all the errors in the back ?
        # 5710  # image 4
        # 34181  # image 1
        # 2586  # image 0 and 1  # VERY interesting case
        #         """
        #         Hypotheses:
        #         Note the images-mappings are mainly correct.

        #         Maybe it is a Image-Pixel mapping problem
        #         The errors are unlikely to come from the projection, so they arise from later processing.

        #             - Area 2 seems worse off than other, is that correct ? Not only... but errors in the office in Area 2 are
        #             worse than other areas...
        #             - global point indexing is too large and some points have the same point_index ?
        #             - +1/-1 index offset when slicing FWD* ?
        #             - get and visualize the point whose mapping creates errors ? See if it the last one, or the first, or
        #             anything noteworthy
        #             - Some points were TOO close to the camera at projection time and created unrealistically large masks ?
        #             - point_index batch reindexing problem check that torch_geometric batch does what we want with the indeices

        #         """
        #         centre = self._centres_for_sampling[centre_idx]

        #         print(f"centre_idx = {centre_idx}")
        #         print(f"i_area = {centre[3].int()}")

        i_area = centre[3].int()
        area_data = self._datas[i_area]
        sphere_sampler = cT.SphereSampling(self._radius, centre[:3], align_origin=False)
        return i_area, sphere_sampler(area_data)

    def _load_data(self, path):
        """
        Initializes the self._datas, self._images and self._mappings which hold 
        all the preprocessed multimodal data in memory. Also initializes the
        sphere sampling centers and per-area KDTrees.
        
        Overwrites the S3DISOriginalFusedMM._load_data()
        """
        self._datas, self._images, self._mappings = torch.load(path)

        if not isinstance(self._datas, list):
            self._datas = [self._datas]
        if not isinstance(self._images, list):
            self._images = [self._images]
        if not isinstance(self._mappings, list):
            self._mappings = [self._mappings]

        if self._sample_per_epoch > 0:
            self._centres_for_sampling = []
            for i, data in enumerate(self._datas):
                # Just to make we don't have some out-of-date data in there
                assert not hasattr(data, cT.SphereSampling.KDTREE_KEY)
                low_res = self._grid_sphere_sampling(data.clone())
                centres = torch.empty((low_res.pos.shape[0], 5), dtype=torch.float)
                centres[:, :3] = low_res.pos
                centres[:, 3] = i
                centres[:, 4] = low_res.y
                self._centres_for_sampling.append(centres)
                tree = KDTree(np.asarray(data.pos), leaf_size=10)
                setattr(data, cT.SphereSampling.KDTREE_KEY, tree)

            self._centres_for_sampling = torch.cat(self._centres_for_sampling, 0)
            uni, uni_counts = np.unique(np.asarray(self._centres_for_sampling[:, -1]),
                                        return_counts=True)
            uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
            self._label_counts = uni_counts / np.sum(uni_counts)
            self._labels = uni

        else:
            # Save the area id in _test_spheres so we can recover multimodal
            # mappings after GridSphereSampling
            for i_area in range(len(self._datas)):
                self._datas[i_area].area_id = i_area
            grid_sampler = cT.GridSphereSampling(self._radius, self._radius, center=False)
            self._test_spheres = grid_sampler(self._datas)


################################################################################
#                          S3DIS TP3D Dataset Wrapper                          #
################################################################################

class S3DISFusedDataset(BaseDatasetMM):
    """ Wrapper around S3DISSphereMM that creates train and test datasets.

    http://buildingparser.stanford.edu/dataset.html

    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain

            - dataroot
            - fold: test_area parameter
            - pre_collate_transform
            - train_transforms
            - test_transforms
    """

    INV_OBJECT_LABEL = INV_OBJECT_LABEL

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        sampling_format = dataset_opt.get('sampling_format', 'sphere')
        assert sampling_format == 'sphere', f"Only sampling format 'sphere' is supported."

        self.train_dataset = S3DISSphereMM(
            self._data_path,
            sample_per_epoch=3000,
            test_area=self.dataset_opt.fold,
            split="train",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.train_transform,
            pre_transform_image=self.pre_transform_image,
            transform_image=self.train_transform_image,
        )

        self.val_dataset = S3DISSphereMM(
            self._data_path,
            sample_per_epoch=-1,
            test_area=self.dataset_opt.fold,
            split="val",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.val_transform,
            pre_transform_image=self.pre_transform_image,
            transform_image=self.val_transform_image,
        )
        
        # self.trainval_dataset = S3DISSphereMM(
        #     self._data_path,
        #     sample_per_epoch=3000,
        #     test_area=self.dataset_opt.fold,
        #     split="trainval",
        #     pre_collate_transform=self.pre_collate_transform,
        #     transform=self.train_transform,
        #     pre_transform_image=self.pre_transform_image,
        #     transform_image=self.train_transform_image,
        # )

        self.test_dataset = S3DISSphereMM(
            self._data_path,
            sample_per_epoch=-1,
            test_area=self.dataset_opt.fold,
            split="test",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.test_transform,
            pre_transform_image=self.pre_transform_image,
            transform_image=self.test_transform_image,
        )

        if dataset_opt.class_weight_method:
            self.train_dataset = add_weights(
                self.train_dataset, True, dataset_opt.class_weight_method)

    @property
    def test_data(self):
        return self.test_dataset[0].raw_test_data

    @staticmethod
    def to_ply(pos, label, file):
        """ Allows to save s3dis predictions to disk using s3dis color scheme

        Parameters
        ----------
        pos : torch.Tensor
            tensor that contains the positions of the points
        label : torch.Tensor
            predicted label
        file : string
            Save location
        """
        to_ply(pos, label, file)

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        from torch_points3d.metrics.s3dis_tracker import S3DISTracker

        return S3DISTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)