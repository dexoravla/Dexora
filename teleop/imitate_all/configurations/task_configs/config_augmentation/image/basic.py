from torchvision.transforms import v2

color_transforms_1 = v2.Compose(
    [
        v2.ColorJitter(brightness=(0.5, 1.5)),
        v2.ColorJitter(contrast=(0.5, 1.5)),
        v2.RandomAdjustSharpness(sharpness_factor=2, p=1),
    ]
)

color_transforms_2 = v2.Compose(
    [
        v2.RandomPhotometricDistort(p=0.13),
        v2.RandomAdjustSharpness(sharpness_factor=2, p=0.13),
        v2.RandomEqualize(p=0.13),
        v2.RandomErasing(p=0.13, scale=(0.02, 0.02), ratio=(0.3, 0.3), value='random'),
        v2.RandomGrayscale(p=0.13),
    ]
)