import math
import cv2
import numpy as np
import ncnn


class NcnnCrackSegmenterV2:
    """
    NCNN wrapper optimized for Pi 4B.

    Important change from v1:
      - inference still runs at 160x160 (or configured model size)
      - returns the SMALL mask directly
      - does NOT resize the mask to 640x480

    The original 640x480 frame is analyzed only inside crack ROIs by
    crack_measurement_v2.py. This avoids full-frame high-resolution
    morphology/skeleton work.
    """

    def __init__(self, cfg):
        mc = cfg["ncnn_model"]
        self.input_blob = mc.get("input_blob", "in0")
        self.output_blob = mc.get("output_blob", "out0")
        self.in_w = int(mc.get("input_width", 160))
        self.in_h = int(mc.get("input_height", 160))
        self.output_is_logits = bool(mc.get("output_is_logits", True))
        self.prob_threshold = float(mc.get("probability_threshold", 0.5))

        self.net = ncnn.Net()
        self.net.opt.num_threads = int(mc.get("num_threads", 4))
        self.net.opt.use_packing_layout = True

        # Optional ARM memory/bandwidth optimizations. Unsupported options are
        # simply ignored so this remains portable across ncnn builds.
        for attr, value in (
            ("use_fp16_storage", bool(mc.get("use_fp16_storage", True))),
            ("use_fp16_packed", bool(mc.get("use_fp16_packed", True))),
            ("use_fp16_arithmetic", bool(mc.get("use_fp16_arithmetic", False))),
        ):
            try:
                if hasattr(self.net.opt, attr):
                    setattr(self.net.opt, attr, value)
            except Exception:
                pass

        ret_p = self.net.load_param(mc["param_path"])
        ret_b = self.net.load_model(mc["bin_path"])
        if ret_p != 0:
            raise RuntimeError(f"NCNN load_param failed ({ret_p}): {mc['param_path']}")
        if ret_b != 0:
            raise RuntimeError(f"NCNN load_model failed ({ret_b}): {mc['bin_path']}")

        if self.output_is_logits:
            p = min(max(self.prob_threshold, 1e-6), 1.0 - 1e-6)
            self.logit_threshold = math.log(p / (1.0 - p))
        else:
            self.logit_threshold = None

        print(f"[INFO] NCNN V2 loaded: {mc['param_path']} + {mc['bin_path']}")
        print(f"[INFO] blobs: {self.input_blob} -> {self.output_blob}, input={self.in_w}x{self.in_h}")
        print("[INFO] V2 mode: small segmentation mask + original-frame ROI measurement")

    def predict_mask_small(self, frame_bgr):
        """Return uint8 0/255 segmentation mask at model resolution only."""
        resized = cv2.resize(
            frame_bgr,
            (self.in_w, self.in_h),
            interpolation=cv2.INTER_AREA,
        )

        mat_in = ncnn.Mat.from_pixels(
            resized,
            ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            self.in_w,
            self.in_h,
        )
        mat_in.substract_mean_normalize(
            [0.0, 0.0, 0.0],
            [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0],
        )

        ex = self.net.create_extractor()
        try:
            ex.set_light_mode(True)
        except Exception:
            pass

        ret = ex.input(self.input_blob, mat_in)
        if ret != 0:
            raise RuntimeError(f"NCNN input failed: ret={ret}, blob={self.input_blob}")

        ret, mat_out = ex.extract(self.output_blob)
        if ret != 0:
            raise RuntimeError(f"NCNN extract failed: ret={ret}, blob={self.output_blob}")

        arr = np.asarray(mat_out, dtype=np.float32)
        arr = np.squeeze(arr)
        if arr.size != self.in_w * self.in_h:
            raise RuntimeError(f"Unexpected NCNN output shape={arr.shape}, size={arr.size}")
        arr = arr.reshape(self.in_h, self.in_w)

        # For logits, sigmoid(x) >= p is exactly equivalent to
        # x >= log(p/(1-p)). Avoiding exp() saves a little CPU time.
        if self.output_is_logits:
            mask_small = arr >= self.logit_threshold
        else:
            mask_small = arr >= self.prob_threshold

        return mask_small.astype(np.uint8) * 255
