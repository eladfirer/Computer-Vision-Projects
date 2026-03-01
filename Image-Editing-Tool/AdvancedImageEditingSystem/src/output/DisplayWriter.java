package output;

import image_editor.image.Image;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

public class DisplayWriter implements OutputWriter {

    public DisplayWriter() {
    }

    // used AI assistance in order to display the image
    @Override
    public void outputImage(Image image) {
        int W = image.getWidth();
        int H = image.getHeight();

        BufferedImage buf = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                Color c = image.getPixel(y, x);
                buf.setRGB(x, y, c.getRGB());
            }
        }

        // 2) Get the screen (usable) size
        Rectangle screen = GraphicsEnvironment
                .getLocalGraphicsEnvironment()
                .getMaximumWindowBounds();
        int maxW = screen.width  - 50;  // leave some margin
        int maxH = screen.height - 50;

        // 3) Compute scale factor (≤1.0) to fit image into the screen
        double scale = Math.min(1.0, Math.min((double)maxW / W, (double)maxH / H));
        int dispW = (int)(W * scale);
        int dispH = (int)(H * scale);

        // 4) If scaling is needed, create a scaled instance
        ImageIcon icon = (scale < 1.0)
                ? new ImageIcon(buf.getScaledInstance(dispW, dispH, java.awt.Image.SCALE_SMOOTH))
                : new ImageIcon(buf);

        // 5) Show it on the EDT
        SwingUtilities.invokeLater(() -> {
            JLabel lbl = new JLabel(icon);
            JScrollPane scroll = new JScrollPane(lbl);

            JFrame frame = new JFrame("Edited Image");
            frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            frame.getContentPane().add(scroll);
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }
}
