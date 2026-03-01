package operations;

import image_editor.image.Image;
import image_editor.image.ImageUtils;

import java.awt.Color;


public class BoxBlurOperation implements Operation {
    private final int fullW, fullH;

    public BoxBlurOperation(int width, int height) {
        this.fullW = width;
        this.fullH = height;
    }

    // Used Ai assistance in order to make the box blur algorithm.
    /**
     * lets the make the box blur kernel algorithm.
     * we going to make this using shrink the kernel in edge cases.
     * we should optimize runtime complexity, so first off all make a sums area table of the pixelArray image.
     * we going to use it to check for each pixel its new value.
     * then calculate the new pixel for each value in the array.
     */

    @Override
    public void activateOnImage(Image image) {
        int kernelHalfW = fullW  / 2;
        int kernelHalfH = fullH / 2;
        int imageH = image.getHeight();
        int imageW = image.getWidth();

        long[][][] sums = ImageUtils.getInstance()
                .computeSummedAreaTables(image);
        long[][] sumR = sums[0], sumG = sums[1], sumB = sums[2];

        for (int y = 0; y < imageH; y++) {
            int yUp = Math.max(0, y - kernelHalfH) + 1;
            int yDown = Math.min(imageH -1, y + kernelHalfH) + 1;
            for (int x = 0; x < imageW; x++) {
                int xLeft = Math.max(0, x - kernelHalfW) + 1;
                int xRight = Math.min(imageW -1, x + kernelHalfW) + 1;
                int area = (yDown - yUp + 1) * (xRight - xLeft + 1);

                long rSum = sumR[yDown][xRight] - sumR[yUp -1][xRight] - sumR[yDown][xLeft -1] + sumR[yUp -1][xLeft -1];
                long gSum = sumG[yDown][xRight] - sumG[yUp -1][xRight] - sumG[yDown][xLeft -1] + sumG[yUp -1][xLeft -1];
                long bSum = sumB[yDown][xRight] - sumB[yUp -1][xRight] - sumB[yDown][xLeft -1] + sumB[yUp -1][xLeft -1];

                // we add area/2 to get the closest integer to rSum/area instead of rounding down.
                int r = ImageUtils.getInstance().clamp((int)((rSum + area/2) / area));
                int g = ImageUtils.getInstance().clamp((int)((gSum + area/2) / area));
                int b = ImageUtils.getInstance().clamp((int)((bSum + area/2) / area));

                image.setPixel(y, x, new Color(r, g, b));
            }
        }

    }
}
