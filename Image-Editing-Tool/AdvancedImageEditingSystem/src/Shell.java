import program.Program;
import program.ProgramFactory;
import utils.Constants;

import java.io.IOException;

public class Shell {

    public static void main(String[] args) {
        if (args.length != 2 || !args[0].equals("--config")) {
            throw new IllegalArgumentException("Usage: edit-image --config <path_to_json_file>");
        }

        String configPath = args[1];
        try {
            // first parameter allows us to support different Program structures in the future (e.g., 'image_editor', 'video_editor', etc.).
            // third parameter allows us to support different input options in the future (e.g., 'json', 'console', etc.).
            Program program = ProgramFactory.createProgram(Constants.IMAGE_EDITOR,configPath,Constants.JSON);
            program.execute();
        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }
}
