package operations;

import image_editor.image.Image;
import image_editor.image.ImageUtils;
import utils.Constants;

import java.awt.*;

public class SharpenOperation implements Operation {
    private final double alpha;

    public SharpenOperation(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public void activateOnImage(Image image) {
        Image blurred = new Image(image);

        new BoxBlurOperation(Constants.SHARPEN_KERNEL_SIZE, Constants.SHARPEN_KERNEL_SIZE)
                .activateOnImage(blurred);

        int H = image.getHeight(), W = image.getWidth();
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                Color c0 = image.getPixel(y, x);
                Color cb = blurred.getPixel(y, x);

                int dr = c0.getRed()   - cb.getRed();
                int dg = c0.getGreen() - cb.getGreen();
                int db = c0.getBlue()  - cb.getBlue();

                int r = ImageUtils.getInstance().clamp((int)Math.round(c0.getRed()   + alpha*dr));
                int g = ImageUtils.getInstance().clamp((int)Math.round(c0.getGreen() + alpha*dg));
                int b = ImageUtils.getInstance().clamp((int)Math.round(c0.getBlue()  + alpha*db));

                image.setPixel(y, x, new Color(r, g, b));
            }
        }

    }
}
