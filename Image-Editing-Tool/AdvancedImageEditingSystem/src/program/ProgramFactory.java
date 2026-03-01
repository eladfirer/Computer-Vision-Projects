package program;

import image_editor.ImageEditor;
import utils.Constants;

public class ProgramFactory {
    public static Program createProgram(String programType, String filePath, String inputType) {
        switch (programType) {
            case Constants.IMAGE_EDITOR:
                return new ImageEditor(filePath, inputType);
                default:
                    throw new IllegalArgumentException("Unsupported program type: " + programType);
        }
    }
}
