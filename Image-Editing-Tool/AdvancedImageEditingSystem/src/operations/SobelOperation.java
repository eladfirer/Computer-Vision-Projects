package operations;

import image_editor.image.Image;
import image_editor.image.ImageUtils;

import java.awt.*;

public class SobelOperation implements Operation {
    private final double threshold;

    public SobelOperation(double threshold) {
        this.threshold = threshold;
    }

    @Override
    public void activateOnImage(Image image) {
        int[][] gray = ImageUtils.getInstance().RGB2grayscale(image.getPixelArray());


        int[][] sobelX = {
                { -1,  0,  1 },
                { -2,  0,  2 },
                { -1,  0,  1 }
        };

        int[][] sobelY = ImageUtils.getInstance().transpose(sobelX);

        int[][] gx = ImageUtils.getInstance().convolveWithMirror(gray, sobelX);
        int[][] gy = ImageUtils.getInstance().convolveWithMirror(gray, sobelY);

        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                double mag = Math.hypot(gx[y][x], gy[y][x]);
                Color c = (mag < threshold)
                        ? Color.WHITE
                        : Color.black;
                image.setPixel(y, x, c);
            }
        }
    }
}
