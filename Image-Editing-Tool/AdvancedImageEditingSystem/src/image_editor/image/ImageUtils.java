package image_editor.image;

import java.awt.Color;

public class ImageUtils {
    private static ImageUtils instance = null;

    private ImageUtils() { }

    public static ImageUtils getInstance() {
        if (instance == null) {
            instance = new ImageUtils();
        }
        return instance;
    }

    public long[][][] computeSummedAreaTables(Image img) {
        int H = img.getHeight(), W = img.getWidth();
        long[][] sumR = new long[H + 1][W + 1];
        long[][] sumG = new long[H + 1][W + 1];
        long[][] sumB = new long[H + 1][W + 1];

        for (int y = 1; y <= H; y++) {
            for (int x = 1; x <= W; x++) {
                Color c = img.getPixel(y - 1, x - 1);
                sumR[y][x] = c.getRed()
                        + sumR[y - 1][x]
                        + sumR[y][x - 1]
                        - sumR[y - 1][x - 1];
                sumG[y][x] = c.getGreen()
                        + sumG[y - 1][x]
                        + sumG[y][x - 1]
                        - sumG[y - 1][x - 1];
                sumB[y][x] = c.getBlue()
                        + sumB[y - 1][x]
                        + sumB[y][x - 1]
                        - sumB[y - 1][x - 1];
            }
        }

        return new long[][][]{ sumR, sumG, sumB };
    }

    public int clamp(int v) {
        return v < 0 ? 0 : Math.min(v, 255);
    }

    public int[][] RGB2grayscale(Color[][] colorImage) {
        int H = colorImage.length, W = colorImage[0].length;
        int[][] gray = new int[H][W];
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                Color c = colorImage[y][x];
                gray[y][x] = (int) Math.round(
                        c.getRed()*0.299 + c.getGreen()*0.587 + c.getBlue()*0.114
                );
            }
        }
        return gray;
    }


    public int[][] transpose(int[][] k) {
        int n = k.length;
        int[][] t = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                t[i][j] = k[j][i];
            }
        }
        return t;
    }

    // Used Ai assistance in order to make the convolution function.
    public int[][] convolveWithMirror(int[][] matrix, int[][] kernel) {
        int H = matrix.length, W = matrix[0].length;
        int k2 = kernel.length/2;
        int[][] out = new int[H][W];

        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int sum = 0;
                for (int ky = -k2; ky <= k2; ky++) {
                    for (int kx = -k2; kx <= k2; kx++) {
                        int yy = mirrorIndex(y + ky, H);
                        int xx = mirrorIndex(x + kx, W);
                        sum += matrix[yy][xx] * kernel[ky + k2][kx + k2];
                    }
                }
                out[y][x] = sum;
            }
        }
        return out;
    }

    private int mirrorIndex(int p, int length) {
        if (p < 0) {
            p = -p;
        }
        int period = 2 * length - 2;
        p = p % period;
        if (p >= length) {
            p = period - p;
        }
        return p;
    }


    // used Ai assitance in order to change rgb to hsl
    // prompt: make a function that changes rgb value into hsl values
    public float[] rgbToHsl(Color c) {
        float r = c.getRed()/255f, g = c.getGreen()/255f, b = c.getBlue()/255f;
        float max = Math.max(r, Math.max(g,b)), min = Math.min(r, Math.min(g,b));
        float h, s, l = (max + min)/2;

        if (max == min) {
            h = 0;
            s = 0;
        } else {
            float d = max - min;
            s = l > 0.5f ? d/(2 - max - min) : d/(max + min);
            if      (max == r) h = ((g - b)/d + (g<b?6:0))*60;
            else if (max == g) h = ((b - r)/d + 2)*60;
            else               h = ((r - g)/d + 4)*60;
        }
        return new float[]{h, s, l};
    }

    // used Ai assitance in order to change hsl to rgb
    // prompt: make a function that changes hsl value into rgb values
    public Color hslToRgb(float[] hsl) {
        float h = hsl[0]/360f, s = hsl[1], l = hsl[2];
        float r, g, b;
        if (s == 0) {
            r = g = b = l;
        } else {
            float q = l < 0.5f ? l * (1 + s) : l + s - l*s;
            float p = 2*l - q;
            r = hue2rgb(p, q, h + 1f/3f);
            g = hue2rgb(p, q, h);
            b = hue2rgb(p, q, h - 1f/3f);
        }
        return new Color(clamp((int)(r*255)), clamp((int)(g*255)), clamp((int)(b*255)));
    }

    private float hue2rgb(float p, float q, float t) {
        if (t < 0)   t += 1;
        if (t > 1)   t -= 1;
        if (t < 1f/6f)   return p + (q-p)*6*t;
        if (t < 1f/2f)   return q;
        if (t < 2f/3f)   return p + (q-p)*(2f/3f - t)*6;
        return p;
    }
}
