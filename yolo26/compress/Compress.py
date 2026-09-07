from compress import GM
import torch.nn.utils.prune as prune
import os
import time
import torch.nn as nn
import torch
from ultralytics.nn.modules import *
import yaml
from ultralytics import YOLO


class PruneHandler():
    def __init__(self, model, compression_ratio, method, cfg_output_path, prune_type='ALL', align=8):
        self.model = model
        self.ckpt = model.ckpt['model']
        self.model.cpu()
        self.cr = compression_ratio
        self.method = method
        self.cfg_output_path = cfg_output_path
        self.model.to('cpu')  # cuda cannot convert to numpy
        self.remain_index_out = {}
        self.prune_type = prune_type
        # 남기는 채널 수를 항상 align 의 배수로 맞춘다(양자화/NPU 커널 정렬 안정화).
        # align=1 이면 정렬 없이 기존처럼 비율만 적용.
        self.align = max(1, int(align))

    def _amt(self, module):
        """이 conv 에서 잘라낼 output 채널 개수(int)를 반환.

        남길 채널 수 keep 을 self.align 의 배수로 반올림해, 프루닝 후 채널이
        항상 정렬된 값(예: 8의 배수)이 되도록 한다. torch 의 ln_structured /
        GMStructured 모두 amount 가 int 면 '자를 채널 개수'로 해석한다.
        """
        num = module.weight.shape[0]
        keep = int(round(num * (1.0 - self.cr) / self.align)) * self.align
        keep = max(self.align, keep)   # 최소 한 그룹은 남김
        keep = min(keep, num)          # num 초과 방지
        keep = max(1, keep)            # 최소 1채널 보장(아주 작은 conv 보호)
        return max(0, num - keep)

    def prune(self):
        backbone_layers = [str(i) for i in range(11)]
        detect_layers = ['23']

        def in_layers(name, layers):
            return any(name == n or name.startswith(n + '.') for n in layers)

        if self.method == 'GM':
            if self.prune_type == 'ALL':
                for name, module in self.model.model.model.named_modules():
                    if not in_layers(name, detect_layers):
                        if isinstance(module, nn.Conv2d):
                            GM.gm_structured(module, name='weight', amount=self._amt(module), dim=0)
                            mask = torch.where(torch.norm(module.weight_mask, 1, dim=(1, 2, 3)) != 0, 1, 0)
                            prune.remove(module, 'weight')
                        if isinstance(module, nn.BatchNorm2d):
                            prune.custom_from_mask(module, name='weight', mask=mask)
                            prune.custom_from_mask(module, name='bias', mask=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')
            elif self.prune_type == 'H':
                for name, module in self.model.model.model.named_modules():
                    if not in_layers(name, backbone_layers + detect_layers):
                        if isinstance(module, torch.nn.Conv2d):
                            GM.gm_structured(module, name='weight', amount=self._amt(module), dim=0)
                            mask = torch.where(torch.norm(module.weight_mask, 1, dim=(1, 2, 3)) != 0, 1, 0)
                            prune.remove(module, 'weight')
                        elif isinstance(module, torch.nn.BatchNorm2d):
                            prune.custom_from_mask(module, name='weight', mask=mask)
                            prune.custom_from_mask(module, name='bias', mask=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')
            elif self.prune_type == 'B':
                for name, module in self.model.model.model.named_modules():
                    if in_layers(name, backbone_layers):
                        if isinstance(module, torch.nn.Conv2d):
                            GM.gm_structured(module, name='weight', amount=self._amt(module), dim=0)
                            mask = torch.where(torch.norm(module.weight_mask, 1, dim=(1, 2, 3)) != 0, 1, 0)
                            prune.remove(module, 'weight')
                        elif isinstance(module, torch.nn.BatchNorm2d):
                            prune.custom_from_mask(module, name='weight', mask=mask)
                            prune.custom_from_mask(module, name='bias', mask=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')

        elif self.method == 'L1':
            if self.prune_type == 'ALL':
                for name, module in self.model.model.model.named_modules():
                    if not in_layers(name, detect_layers):
                        if isinstance(module, nn.Conv2d):
                            prune.ln_structured(module, name='weight', amount=self._amt(module), n=1, dim=0)
                            mask = torch.where(torch.norm(module.weight_mask, p=2, dim=(1, 2, 3)) != 0, 1, 0)
                            prune.remove(module, 'weight')
                        if isinstance(module, nn.BatchNorm2d):
                            prune.custom_from_mask(module, name='weight', mask=mask)
                            prune.custom_from_mask(module, name='bias', mask=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')
            elif self.prune_type == 'H':
                for name, module in self.model.model.model.named_modules():
                    if not in_layers(name, backbone_layers + detect_layers):
                        if isinstance(module, torch.nn.Conv2d):
                            prune.ln_structured(module, name='weight', amount=self._amt(module), n=1, dim=0)
                            mask = torch.where(torch.norm(module.weight_mask, p=2, dim=(1, 2, 3)) != 0, 1, 0)
                            prune.remove(module, 'weight')
                        elif isinstance(module, torch.nn.BatchNorm2d):
                            prune.custom_from_mask(module, name='weight', mask=mask)
                            prune.custom_from_mask(module, name='bias', mask=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')
            elif self.prune_type == 'B':
                for name, module in self.model.model.model.named_modules():
                    if in_layers(name, backbone_layers):
                        if isinstance(module, torch.nn.Conv2d):
                            prune.ln_structured(module, name='weight', amount=self._amt(module), n=1, dim=0)
                            mask = torch.where(torch.norm(module.weight_mask, p=2, dim=(1, 2, 3)) != 0, 1, 0)
                            prune.remove(module, 'weight')
                        elif isinstance(module, torch.nn.BatchNorm2d):
                            prune.custom_from_mask(module, name='weight', mask=mask)
                            prune.custom_from_mask(module, name='bias', mask=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')

        elif self.method == 'L2':
            if self.prune_type == 'ALL':
                for name, module in self.model.model.model.named_modules():
                    if not in_layers(name, detect_layers):
                        if isinstance(module, nn.Conv2d):
                            prune.ln_structured(module, name='weight', amount=self._amt(module), n=2, dim=0)
                            mask = torch.where(torch.norm(module.weight_mask, p=2, dim=(1, 2, 3)) != 0, 1, 0)
                            prune.remove(module, 'weight')
                        if isinstance(module, nn.BatchNorm2d):
                            prune.custom_from_mask(module, name='weight', mask=mask)
                            prune.custom_from_mask(module, name='bias', mask=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')
            elif self.prune_type == 'H':
                for name, module in self.model.model.model.named_modules():
                    if not in_layers(name, backbone_layers + detect_layers):
                        if isinstance(module, torch.nn.Conv2d):
                            prune.ln_structured(module, name='weight', amount=self._amt(module), n=2, dim=0)
                            mask = torch.where(torch.norm(module.weight_mask, p=2, dim=(1, 2, 3)) != 0, 1, 0)
                            prune.remove(module, 'weight')
                        elif isinstance(module, torch.nn.BatchNorm2d):
                            prune.custom_from_mask(module, name='weight', mask=mask)
                            prune.custom_from_mask(module, name='bias', mask=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')
            elif self.prune_type == 'B':
                for name, module in self.model.model.model.named_modules():
                    if in_layers(name, backbone_layers):
                        if isinstance(module, torch.nn.Conv2d):
                            prune.ln_structured(module, name='weight', amount=self._amt(module), n=2, dim=0)
                            mask = torch.where(torch.norm(module.weight_mask, p=2, dim=(1, 2, 3)) != 0, 1, 0)
                            prune.remove(module, 'weight')
                        elif isinstance(module, torch.nn.BatchNorm2d):
                            prune.custom_from_mask(module, name='weight', mask=mask)
                            prune.custom_from_mask(module, name='bias', mask=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')

    def reconstruct(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.training = False

        detect_in_channels = []
        concat = {}
        remain_out_channels = [0, 1, 2]
        for name, module in self.model.model.model.named_modules():
            if isinstance(module, Conv):
                if name in ['0', '1', '3', '5', '7', '17', '20']:
                    num = int(name)
                    offset = self.model.model.model[num].conv.weight.shape[0]
                    remain_out_channels = module.recon(remain_out_channels)
                    if name in ['17', '20']:
                        concat[name] = [remain_out_channels, offset]

            elif isinstance(module, C3k2):
                first_m = module.m[0]

                if hasattr(first_m, "cv1") and hasattr(first_m, "cv3"):
                    c3k = True
                else:
                    c3k = False
                num = int(name)
                offset = self.model.model.model[num].cv2.conv.weight.shape[0]
                remain_out_channels = module.recon(remain_out_channels, c3k)
                if name in ['16', '19', '22']:
                    detect_in_channels.append(remain_out_channels)
                elif name in ['4', '6', '13']:
                    concat[name] = [remain_out_channels, offset]

            elif isinstance(module, SPPF):
                num = int(name)
                offset = self.model.model.model[num].cv2.conv.weight.shape[0]
                remain_out_channels = module.recon(remain_out_channels)
                concat[name] = [remain_out_channels, offset]

            elif isinstance(module, C2PSA):
                num = int(name)
                offset = self.model.model.model[num].cv2.conv.weight.shape[0]
                remain_out_channels = module.recon(remain_out_channels)
                concat[name] = [remain_out_channels, offset]

            elif isinstance(module, Detect):
                remain_out_channels = module.recon(detect_in_channels)

            elif isinstance(module, Concat):
                if name == '12':
                    concat['6'][0] = [x + concat['10'][1] for x in concat['6'][0]]
                    remain_out_channels = concat['10'][0] + concat['6'][0]
                elif name == '15':
                    concat['4'][0] = [x + concat['13'][1] for x in concat['4'][0]]
                    remain_out_channels = concat['13'][0] + concat['4'][0]
                elif name == '18':
                    concat['13'][0] = [x + concat['17'][1] for x in concat['13'][0]]
                    remain_out_channels = concat['17'][0] + concat['13'][0]
                elif name == '21':
                    concat['10'][0] = [x + concat['20'][1] for x in concat['10'][0]]
                    remain_out_channels = concat['20'][0] + concat['10'][0]

    def _exact_channel_cfg(self, module, attrs=None):
        modules = dict(module.named_modules())
        conv = {}
        conv2d = {}

        for name, submodule in module.named_modules():
            if isinstance(submodule, Conv):
                conv[name or "_self"] = [
                    int(submodule.conv.in_channels),
                    int(submodule.conv.out_channels),
                    int(submodule.conv.groups),
                ]

        for name, submodule in module.named_modules():
            if not isinstance(submodule, nn.Conv2d):
                continue
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent = modules.get(parent_name, module)
            if isinstance(parent, Conv) and name.endswith(".conv"):
                continue
            conv2d[name] = [
                int(submodule.in_channels),
                int(submodule.out_channels),
                int(submodule.groups),
            ]

        cfg = {"exact": True}
        if attrs:
            cfg["attrs"] = {k: int(v) if isinstance(v, (int, bool)) else v for k, v in attrs.items()}
        if conv:
            cfg["conv"] = conv
        if conv2d:
            cfg["conv2d"] = conv2d
        return cfg

    def model_to_yaml(self):
        from_ = -1
        repeats = 1
        yaml_dict = {}
        detect = self.model.model.model[-1]
        yaml_dict["nc"] = detect.nc  # baseline/실제 학습 클래스 수를 그대로 따름 (하드코딩 8 제거)
        yaml_dict["end2end"] = detect.end2end
        yaml_dict["reg_max"] = detect.reg_max
        yaml_dict["exact_channels"] = True
        yaml_dict["scales"] = {'prune': [1, 1, 1024]}
        yaml_dict["backbone"] = []
        yaml_dict["head"] = []

        for name, module in self.model.model.model.named_modules():
            if isinstance(module, Conv):
                if name in ['0', '1', '3', '5', '7', '17', '20']:
                    args = [module.conv.out_channels, module.conv.kernel_size[0], module.conv.stride[0]]
                    layer = [from_, repeats, type(module).__name__, args]
                    if name in ["17", "20"]:
                        yaml_dict["head"].append(layer)
                    else:
                        yaml_dict["backbone"].append(layer)

            elif isinstance(module, C3k2):
                # 직접 전달 하게 함.
                attn = (
                    len(module.m) > 0
                    and isinstance(module.m[0], nn.Sequential)
                    and len(module.m[0]) > 1
                    and module.m[0][1].__class__.__name__ == "PSABlock"
                )
                if name in ['2', '4']:
                    c3k = False
                    e = 0.25
                else:
                    c3k = True
                    e = 0.5               
                args = [
                    module.cv2.conv.out_channels,
                    c3k,
                    e,
                    attn,
                    self._exact_channel_cfg(module, {"c": module.c}),
                ]
                layer = [from_, len(module.m), type(module).__name__, args]
                if name in ['13', '16', '19', '22']:
                    yaml_dict["head"].append(layer)
                else:
                    yaml_dict["backbone"].append(layer)

            
            elif isinstance(module, SPPF):
                k = module.k
                c2 = module.cv2.conv.out_channels  # 실제 pruned 채널 그대로 (max(.,128) 인플레이션 제거)

                args = [c2, k, module.n, module.shortcut, self._exact_channel_cfg(module)]
                layer = [from_, repeats, type(module).__name__, args]
                yaml_dict["backbone"].append(layer)

            
            elif isinstance(module, C2PSA):
                c2 = module.cv2.conv.out_channels  # 실제 pruned 채널 그대로 (max(.,128) 인플레이션 제거)

                args = [c2, module.e, self._exact_channel_cfg(module, {"c": module.c})]
                layer = [from_, len(module.m), type(module).__name__, args]
                yaml_dict["backbone"].append(layer)

            elif isinstance(module, Detect):
                args = [yaml_dict["nc"], self._exact_channel_cfg(module)]
                layer = [[16, 19, 22], repeats, type(module).__name__, args]
                yaml_dict["head"].append(layer)

            elif isinstance(module, Concat):
                args = [1]
                if name == '12':
                    # import pdb; pdb.set_trace()
                    layer = [[from_, 6], repeats, type(module).__name__, args]
                elif name == '15':
                    layer = [[from_, 4], repeats, type(module).__name__, args]
                elif name == '18':
                    layer = [[from_, 13], repeats, type(module).__name__, args]
                elif name == '21':
                    layer = [[from_, 10], repeats, type(module).__name__, args]
                yaml_dict["head"].append(layer)

            elif isinstance(module, nn.Upsample):
                args = ['None', module.scale_factor, module.mode]
                layer = [from_, repeats, "nn." + type(module).__name__, args]
                yaml_dict["head"].append(layer)

        self.model.model.yaml = yaml_dict
        if self.model.ckpt and isinstance(self.model.ckpt.get("model"), nn.Module):
            self.model.ckpt["model"].yaml = yaml_dict

        yaml_str = yaml.safe_dump(yaml_dict, sort_keys=False, allow_unicode=True)
        with open(f'./{self.cfg_output_path}/best_model_prune.yaml', "w") as file:
            file.write(yaml_str)

    def compress_yolo26(self):
        print('Pruning...')
        start = time.time()
        os.makedirs(self.cfg_output_path, exist_ok=True)  # 출력 폴더 자동 생성
        self.prune()
        self.reconstruct()
        self.model_to_yaml()
        for p in self.model.model.parameters():
            if p.dtype.is_floating_point:
                p.requires_grad = True
        self.model.model._use_current_model_for_train = True
        self.model.save(f'./{self.cfg_output_path}/best_model_prune.pt')
        print('Done')
        # import pdb; pdb.set_trace()
        print(f'time : {time.time() - start}')
        return self.model
