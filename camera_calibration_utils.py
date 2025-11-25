import os
import numpy as np
import cv2


class ImageUndistorter():
    def undistort(self, image: np.ndarray):
        pass

    @staticmethod
    def save_image(image: np.ndarray, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(
            path,
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 100],
        )

    def undistort_image_file(self, image_file_path: str):
        image = cv2.imread(image_file_path, cv2.IMREAD_UNCHANGED)
        return self.undistort(image)

    def undistort_image_file_to(self, image_file_path: str, output_file_path: str):
        undistorted_image = self.undistort_image_file(image_file_path=image_file_path)
        self.save_image(undistorted_image, output_file_path)

    def undistort_image(self, image: np.ndarray):
        return self.undistort(image)

    def undistort_image_to(self, image: np.ndarray, output_file_path: str):
        undistorted_image = self.undistort_image(image=image)
        self.save_image(undistorted_image, output_file_path)


class ImageRemap(ImageUndistorter):
    def __init__(self, mapx, mapy, target_wh, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT):
        self.mapx = mapx
        self.mapy = mapy
        self.target_shape = (target_wh[1], target_wh[0])
        self.interpolation = interpolation
        self.borderMode = borderMode

    def remap(self, image):
        assert image.shape[:2] == self.target_shape
        return cv2.remap(
            image,
            self.mapx,
            self.mapy,
            interpolation=self.interpolation,
            borderMode=self.borderMode,
        )

    def undistort(self, image):
        return self.remap(image=image)


class FisheyeCamera:
    @staticmethod
    def parse_intrinsics(intrinsics_to_parse):
        """
        Parameters
        ----------
        intrinsics_to_parse: str as npy file path, list in [W H FX FY CX CY K1 K2 K3 K4]

        """
        if len(intrinsics_to_parse) == 1:
            intrinsics = np.load(intrinsics_to_parse[0], allow_pickle=True).item()
            DIM = intrinsics["DIM"]
            K = intrinsics["K"]
            D = np.reshape(intrinsics["D"], (-1, 1))
        else:
            DIM = intrinsics_to_parse[:2]

            K = np.eye(3)
            # fx
            K[0, 0] = float(intrinsics_to_parse[2])
            # fy
            K[1, 1] = float(intrinsics_to_parse[3])
            # cx
            K[0, 2] = float(intrinsics_to_parse[4])
            # cy
            K[1, 2] = float(intrinsics_to_parse[5])

            D = np.zeros((4, 1))
            for i in range(4):
                D[i, 0] = float(intrinsics_to_parse[i + 6])

        return DIM, K, D


class FisheyeCameraUndistorter(ImageUndistorter):
    def __init__(self, DIM, K, D, balance: float = 0):
        self.DIM = DIM
        self.K = K
        self.D = D
        self.balance = balance

    @staticmethod
    def from_intrinsics(intrinsics, balance: float = 0):
        DIM, K, D = FisheyeCamera.parse_intrinsics(intrinsics)
        return FisheyeCameraUndistorter(DIM=DIM, K=K, D=D, balance=balance)

    def remap_preparation(self, target_wh):
        assert target_wh[0] == self.DIM[0]
        assert target_wh[1] == self.DIM[1]
        scaled_K = self.K
        # scaled_K = self.K * target_wh[0] / self.DIM[0]
        # scaled_K[2][2] = 1.0
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            scaled_K,
            self.D,
            target_wh,
            np.eye(3),
            balance=self.balance,
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            scaled_K,
            self.D,
            np.eye(3),
            new_K,
            target_wh,
            cv2.CV_16SC2
        )

        return map1, map2, new_K

    def get_image_remapper(self, target_wh):
        mapx, mapy, _ = self.remap_preparation(target_wh=target_wh)
        return ImageRemap(mapx=mapx, mapy=mapy, target_wh=target_wh)

    # @staticmethod
    # def do_undistort(img, DIM, K, D, balance: float):
    #     img_dim = img.shape[:2][::-1]
    #
    #     scaled_K = K * img_dim[0] / DIM[0]
    #     scaled_K[2][2] = 1.0
    #     new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    #         scaled_K,
    #         D,
    #         img_dim,
    #         np.eye(3),
    #         balance=balance,
    #     )
    #     map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    #         scaled_K,
    #         D,
    #         np.eye(3),
    #         new_K,
    #         img_dim,
    #         cv2.CV_16SC2
    #     )
    #
    #     undist_image = cv2.remap(
    #         img,
    #         map1,
    #         map2,
    #         interpolation=cv2.INTER_LINEAR,
    #         borderMode=cv2.BORDER_CONSTANT,
    #     )
    #
    #     return undist_image

    def undistort(self, image: np.ndarray):
        image_wh = image.shape[:2][::-1]
        remapper = self.get_image_remapper(
            target_wh=image_wh,
        )

        return remapper.remap(image=image)
