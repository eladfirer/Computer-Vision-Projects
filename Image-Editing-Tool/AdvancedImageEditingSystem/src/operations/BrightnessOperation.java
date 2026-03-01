package operations;
import image_editor.image.Image;
import image_editor.image.ImageUtils;

import java.awt.*;

public class BrightnessOperation implements Operation {
    private final double value;

    public BrightnessOperation(double value) {
        this.value = value;
    }

    @Override
    public void activateOnImage(Image image) {
        int h = image.getHeight(), w = image.getWidth();
        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                Color c = image.getPixel(row, col);
                int r = ImageUtils.getInstance().clamp((int) Math.round(c.getRed()   * this.value));
                int g = ImageUtils.getInstance().clamp((int) Math.round(c.getGreen() * this.value));
                int b = ImageUtils.getInstance().clamp((int) Math.round(c.getBlue()  * this.value));
                image.setPixel(row, col, new Color(r, g, b));
            }
        }
    }

}
