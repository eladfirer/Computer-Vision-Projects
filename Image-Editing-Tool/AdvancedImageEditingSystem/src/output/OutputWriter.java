package output;

import image_editor.image.Image;

// we can add in the future outputVideo and more
public interface OutputWriter {
    void outputImage(Image image) throws OutputException;
}
