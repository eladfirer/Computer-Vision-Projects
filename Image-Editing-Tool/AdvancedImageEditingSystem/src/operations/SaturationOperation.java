package operations;

import image_editor.image.Image;
import image_editor.image.ImageUtils;

import java.awt.Color;

public class SaturationOperation implements Operation {
    private final double factor;

    public SaturationOperation(double factor) {
        this.factor = Math.max(0.0, factor);
    }

    @Override
    public void activateOnImage(Image image) {
        int H = image.getHeight(), W = image.getWidth();

        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                Color c = image.getPixel(y, x);
                float[] hsl = ImageUtils.getInstance().rgbToHsl(c);

                hsl[1] = (float) Math.min(1.0, hsl[1] * factor);

                Color nc =  ImageUtils.getInstance().hslToRgb(hsl);
                image.setPixel(y, x, nc);
            }
        }
    }
}
