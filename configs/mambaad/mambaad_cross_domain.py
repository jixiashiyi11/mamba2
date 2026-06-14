import copy
from configs.mambaad.mambaad_medical import cfg as medical_cfg


class cfg(medical_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        # ==========================================
        # 1. 数据集强行分家 (Train 读工业，Test 读医学)
        # ==========================================
        self.data_train = copy.deepcopy(self.data)
        self.data_train.root = 'data/mvtec'
        self.data_train.meta = 'meta.json'

        self.data_test = copy.deepcopy(self.data)
        self.data_test.root = 'data/medical'
        self.data_test.meta = 'meta.json'

        # ==========================================
        # 2. 扩充提示词词典 (必须包含你想要训练的工业类别)
        # ==========================================
        industry_prompts_normal = {
            'bottle': 'A normal hollow cylindrical anatomical-like structure with smooth continuous walls, regular contour, and no rupture, collapse, or internal opacity.',
            'cable': 'A normal elongated cord-like structure with intact outer covering, parallel internal strands, and no discontinuity, swelling, or surface injury.',
            'capsule': 'A normal oval encapsulated structure with a smooth intact shell, homogeneous appearance, and no fissure, collapse, or surface contamination.',
            'carpet': 'A normal dense fibrous tissue-like surface with uniform texture, continuous coverage, and no focal distortion, erosion, or adherent debris.',
            'grid': 'A normal reticular lattice-like structure with regular spacing, symmetric intersections, and no bending, fragmentation, or architectural distortion.',
            'hazelnut': 'A normal rounded nodular structure with an intact outer shell, smooth contour, and no crack, indentation, or focal discoloration.',
            'leather': 'A normal smooth contiguous dermal-like surface with regular micro-texture, uniform tone, and no laceration, ulceration, or pigmentary change.',
            'metal_nut': 'A normal ring-shaped rigid structure with a clear central lumen, regular inner ridges, and no fracture, deformation, or erosive defect.',
            'pill': 'A normal compact tablet-like structure with smooth margins, uniform density, and no chipping, fissure, or surface irregularity.',
            'screw': 'A normal rigid elongated structure with regular spiral protrusions, preserved alignment, and no breakage, torsion, or surface erosion.',
            'tile': 'A normal planar plate-like surface with homogeneous appearance, sharp boundaries, and no fissure, chipping, or focal staining.',
            'toothbrush': 'A normal elongated handle-like structure with a uniform tufted filamentous end, preserved alignment, and no detachment, distortion, or contamination.',
            'transistor': 'A normal small lobulated device-like structure with a compact central body and slender linear extensions, showing preserved integrity and no bending or fragmentation.',
            'wood': 'A normal layered organic matrix-like surface with orderly longitudinal grain, uniform appearance, and no fissure, cavitation, or infiltrative staining.',
            'zipper': 'A normal paired interlocking linear structure with regular repeating elements, preserved alignment, and no separation, breakage, or missing segments.',
        }
        industry_prompts_abnormal = {
            'bottle': 'An abnormal hollow cylindrical anatomical-like structure showing wall rupture, longitudinal cracking, contour collapse, or internal fluid leakage.',
            'cable': 'An abnormal elongated cord-like structure showing sheath disruption, exposed internal strands, focal transection, kinking, or segmental swelling.',
            'capsule': 'An abnormal oval encapsulated structure showing shell fissure, collapse, fragmentation, surface staining, or loss of structural integrity.',
            'carpet': 'An abnormal dense fibrous tissue-like surface showing focal tearing, erosive loss, textural disruption, or adherent foreign material.',
            'grid': 'An abnormal reticular lattice-like structure showing distorted intersections, bending, focal rupture, missing segments, or irregular spacing.',
            'hazelnut': 'An abnormal rounded nodular structure showing shell cracking, surface depression, focal discoloration, or traumatic outer injury.',
            'leather': 'An abnormal dermal-like surface showing deep lacerations, epidermal erosion, focal discoloration, or irregular textural breakdown.',
            'metal_nut': 'An abnormal ring-shaped rigid structure showing fracture, deformation of the central lumen, erosive surface damage, or disrupted inner ridges.',
            'pill': 'An abnormal compact tablet-like structure showing chipping, fissuring, fragmentation, surface contamination, or contour irregularity.',
            'screw': 'An abnormal rigid elongated structure showing broken spiral ridges, bending, torsion, surface abrasion, or partial structural loss.',
            'tile': 'An abnormal planar plate-like surface showing linear fracture, marginal chipping, focal staining, or structural discontinuity.',
            'toothbrush': 'An abnormal elongated handle-like structure showing filament loss, tuft distortion, detachment, contamination, or focal structural damage.',
            'transistor': 'An abnormal small lobulated structure with slender linear extensions showing bent projections, fragmentation, surface injury, or loss of central body integrity.',
            'wood': 'An abnormal layered organic matrix-like surface showing deep fissures, cavitary defects, infiltrative staining, or destructive surface erosion.',
            'zipper': 'An abnormal paired interlocking linear structure showing separation of opposing elements, broken repeating units, misalignment, or segmental loss.',
        }
        industry_class_prompts = {
            'bottle': 'A medical image showing a hollow cylindrical anatomical-like structure with smooth continuous walls and regular contour.',
            'cable': 'A medical image showing an elongated cord-like structure with parallel internal strands and intact outer covering.',
            'capsule': 'A medical image showing an oval encapsulated structure with a smooth shell and homogeneous internal appearance.',
            'carpet': 'A medical image showing a dense fibrous tissue-like surface with continuous coverage and uniform micro-texture.',
            'grid': 'A medical image showing a reticular lattice-like structure with regular spacing and symmetric intersections.',
            'hazelnut': 'A medical image showing a rounded nodular structure with an intact outer shell and smooth contour.',
            'leather': 'A medical image showing a contiguous dermal-like surface with regular micro-texture and uniform tone.',
            'metal_nut': 'A medical image showing a ring-shaped rigid structure with a clear central lumen and regular inner ridges.',
            'pill': 'A medical image showing a compact tablet-like structure with smooth margins and uniform density.',
            'screw': 'A medical image showing a rigid elongated structure with regular spiral protrusions and preserved alignment.',
            'tile': 'A medical image showing a planar plate-like surface with homogeneous appearance and sharp boundaries.',
            'toothbrush': 'A medical image showing an elongated handle-like structure with a uniform tufted filamentous end.',
            'transistor': 'A medical image showing a small lobulated structure with a compact central body and slender linear extensions.',
            'wood': 'A medical image showing a layered organic matrix-like surface with orderly longitudinal grain and uniform appearance.',
            'zipper': 'A medical image showing a paired interlocking linear structure with regular repeating elements and preserved alignment.',
        }

        self.data_train.cls_names = sorted(industry_prompts_normal.keys())

        # 将工业提示词与原有的医学提示词合并
        self.prompt_normal.update(industry_prompts_normal)
        self.prompt_abnormal.update(industry_prompts_abnormal)
        self.class_prompts.update(industry_class_prompts)

        # 把更新后的 prompt 塞回模型 kwargs 里
        self.model.kwargs['prompt_normal'] = self.prompt_normal
        self.model.kwargs['prompt_abnormal'] = self.prompt_abnormal
        self.model.kwargs['class_prompts'] = self.class_prompts
