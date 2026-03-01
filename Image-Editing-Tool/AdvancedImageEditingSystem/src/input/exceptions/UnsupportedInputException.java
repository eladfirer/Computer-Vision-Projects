package input.exceptions;

import java.io.IOException;

public class UnsupportedInputException extends IOException {
    public UnsupportedInputException(String message) {
        super(message);
    }
}
