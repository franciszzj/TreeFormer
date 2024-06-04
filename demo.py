import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image
from network import pvt_cls as TCN

import gradio as gr


def demo(img_path):
    # config
    batch_size = 8
    crop_size = 256
    model_path = '/users/k21163430/workspace/TreeFormer/models/best_model.pth'

    device = torch.device('cuda')

    # prepare model
    model = TCN.pvt_treeformer(pretrained=False)
    model.to(device)
    model.load_state_dict(torch.load(model_path, device))
    model.eval()

    # preprocess
    img = Image.open(img_path).convert('RGB')
    show_img = np.array(img)
    wd, ht = img.size
    st_size = 1.0 * min(wd, ht)
    if st_size < crop_size:
        rr = 1.0 * crop_size / st_size
        wd = round(wd * rr)
        ht = round(ht * rr)
        st_size = 1.0 * min(wd, ht)
        img = img.resize((wd, ht), Image.BICUBIC)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform(img)
    img = img.unsqueeze(0)

    # model forward
    with torch.no_grad():
        inputs = img.to(device)
        crop_imgs, crop_masks = [], []
        b, c, h, w = inputs.size()
        rh, rw = crop_size, crop_size

        for i in range(0, h, rh):
            gis, gie = max(min(h - rh, i), 0), min(h, i + rh)

            for j in range(0, w, rw):
                gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                crop_imgs.append(inputs[:, :, gis:gie, gjs:gje])
                mask = torch.zeros([b, 1, h, w]).to(device)
                mask[:, :, gis:gie, gjs:gje].fill_(1.0)
                crop_masks.append(mask)
        crop_imgs, crop_masks = map(lambda x: torch.cat(
            x, dim=0), (crop_imgs, crop_masks))

        crop_preds = []
        nz, bz = crop_imgs.size(0), batch_size
        for i in range(0, nz, bz):

            gs, gt = i, min(nz, i + bz)
            crop_pred, _ = model(crop_imgs[gs:gt])
            crop_pred = crop_pred[0]

            _, _, h1, w1 = crop_pred.size()
            crop_pred = F.interpolate(crop_pred, size=(
                h1 * 4, w1 * 4), mode='bilinear', align_corners=True) / 16
            crop_preds.append(crop_pred)
        crop_preds = torch.cat(crop_preds, dim=0)

        # splice them to the original size
        idx = 0
        pred_map = torch.zeros([b, 1, h, w]).to(device)
        for i in range(0, h, rh):
            gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
            for j in range(0, w, rw):
                gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                idx += 1
        # for the overlapping area, compute average value
        mask = crop_masks.sum(dim=0).unsqueeze(0)
        outputs = pred_map / mask

        outputs = F.interpolate(outputs, size=(
            h, w), mode='bilinear', align_corners=True)/4
        outputs = pred_map / mask
        model_output = round(torch.sum(outputs).item())

        print("{}: {}".format(img_path, model_output))
        outputs = outputs.squeeze().cpu().numpy()
        outputs = (outputs - np.min(outputs)) / \
            (np.max(outputs) - np.min(outputs))

        show_img = show_img / 255.0
        show_img = show_img * 0.2 + outputs[:, :, None] * 0.8

    return model_output, show_img


if __name__ == "__main__":
    # test
    # img_path = sys.argv[1]
    # demo(img)

    # Launch a gr.Interface
    gr_demo = gr.Interface(fn=demo,
                           inputs=gr.Image(source="upload",
                                           type="filepath",
                                           label="Input Image",
                                           width=768,
                                           height=768,
                                           ),
                           outputs=[
                               gr.Number(label="Predicted Tree Count"),
                               gr.Image(label="Density Map",
                                        width=768,
                                        height=768,
                                        )
                           ],
                           title="TreeFormer",
                           description="TreeFormer is a semi-supervised transformer-based framework for tree counting from a single high resolution image. Upload an image and TreeFormer will predict the number of trees in the image and generate a density map of the trees.",
                           article="This work has been developed a spart of the ReSET project which has received funding from the European Union's Horizon 2020 FET Proactive Programme under grant agreement No 101017857. The contents of this publication are the sole responsibility of the ReSET consortium and do not necessarily reflect the opinion of the European Union.",
                           examples=[
                                 ["./examples/IMG_101.jpg"],
                                 ["./examples/IMG_125.jpg"],
                                 ["./examples/IMG_138.jpg"],
                                 ["./examples/IMG_180.jpg"],
                                 ["./examples/IMG_18.jpg"],
                                 ["./examples/IMG_206.jpg"],
                                 ["./examples/IMG_223.jpg"],
                                 ["./examples/IMG_247.jpg"],
                                 ["./examples/IMG_270.jpg"],
                                 ["./examples/IMG_306.jpg"],
                           ],
                           cache_examples=True,
                           examples_per_page=10,
                           allow_flagging=False,
                           theme=gr.themes.Default(),
                           )
    gr_demo.launch(share=True, server_port=7861, favicon_path="./assets/reset.png")
