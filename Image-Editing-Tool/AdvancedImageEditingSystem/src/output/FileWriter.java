package output;

import image_editor.image.Image;
import utils.Constants;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class FileWriter implements OutputWriter {
    private final String filePath;

    public FileWriter(String filePath) {
        this.filePath = filePath;
    }

    @Override
    public void outputImage(Image image) throws OutputException {
        int w = image.getWidth(), h = image.getHeight();
        BufferedImage buf = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);

        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                Color c = image.getPixel(row, col);
                buf.setRGB(col, row, c.getRGB());
            }
        }

        String fmt = Constants.PNG;
        int dot = filePath.lastIndexOf('.');
        if (dot >= 0 && dot < filePath.length() - 1) {
            fmt = filePath.substring(dot + 1);
        }

        try {
            if (!ImageIO.write(buf, fmt, new File(filePath))) {
                throw new OutputException("No writer found for format: " + fmt);
            }
        } catch (IOException e) {
            throw new OutputException("Failed to save image to " + filePath);
        }
    }
}
