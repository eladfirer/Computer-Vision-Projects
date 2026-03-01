package operations;

import image_editor.image.Image;
import image_editor.image.ImageUtils;

import java.awt.*;

public class ContrastOperation implements Operation {
    private final double value;

    public ContrastOperation(double value) {
        this.value = value;
    }

    @Override
    public void activateOnImage(Image img) {
        int H = img.getHeight();
        int W = img.getWidth();

        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                Color c = img.getPixel(y, x);

                int r = ImageUtils.getInstance().clamp((int)Math.round((c.getRed()   - 128) * value + 128));
                int g = ImageUtils.getInstance().clamp((int)Math.round((c.getGreen() - 128) * value + 128));
                int b = ImageUtils.getInstance().clamp((int)Math.round((c.getBlue()  - 128) * value + 128));

                img.setPixel(y, x, new Color(r, g, b));
            }
        }
    }
}
