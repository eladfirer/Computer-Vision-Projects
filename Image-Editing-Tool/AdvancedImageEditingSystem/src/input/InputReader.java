package input;

import image_editor.image.DataContainer;

import java.io.IOException;

public interface InputReader {
    DataContainer read() throws IOException;
}
