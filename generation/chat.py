from pathlib import Path
import sys


# ==========================================================
# Project root
# ==========================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ==========================================================
# Geniee imports
# ==========================================================

from generation.generate import load_geniee


# ==========================================================
# Configuration
# ==========================================================

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "geniee_epoch_5.pt"
)


# ==========================================================
# Geniee Chat
# ==========================================================

class GenieeChat:
    """
    Interactive command-line chatbot powered by Geniee.
    """

    def __init__(
        self,
        generator
    ):

        self.generator = generator

    # ======================================================
    # Generate response
    # ======================================================

    # def respond(
    #     self,
    #     prompt
    # ):
    #     """
    #     Generate a response for the supplied prompt.
    #     """

    #     if not isinstance(
    #         prompt,
    #         str
    #     ):

    #         raise TypeError(
    #             "prompt must be a string"
    #         )

    #     prompt = prompt.strip()

    #     if not prompt:

    #         return ""

    #     response = (
    #         self.generator.generate(
    #             prompt,
    #             max_new_tokens=50,
    #             temperature=0.8,
    #             top_k=20,
    #             top_p=0.9,
    #             repetition_penalty=1.1,
    #             do_sample=True
    #         )
    #     )

    #     return response
    def respond(
        self,
        prompt
    ):
        """
        Generate only the newly generated response.
        """

        if not isinstance(
            prompt,
            str
        ):

            raise TypeError(
                "prompt must be a string"
            )

        prompt = prompt.strip()

        if not prompt:

            return ""

        response = (
            self.generator.generate(
                prompt,
                max_new_tokens=50,
                temperature=0.8,
                top_k=20,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                return_full_text=False
            )
        )

        return response.strip()

    # ======================================================
    # Start chat
    # ======================================================

    def start(self):

        print()
        print(
            "=" * 60
        )

        print(
            "GENIEE CHAT"
        )

        print(
            "=" * 60
        )

        print()
        print(
            "Type 'exit' or 'quit' to stop."
        )

        print(
            "Type 'clear' to reset the conversation."
        )

        print()

        conversation = []

        while True:

            try:

                user_input = input(
                    "You: "
                )

            except (
                KeyboardInterrupt,
                EOFError
            ):

                print()
                print(
                    "Exiting Geniee..."
                )

                break

            user_input = (
                user_input.strip()
            )

            # --------------------------------------------------
            # Empty input
            # --------------------------------------------------

            if not user_input:

                continue

            # --------------------------------------------------
            # Exit
            # --------------------------------------------------

            if user_input.lower() in {
                "exit",
                "quit"
            }:

                print()
                print(
                    "Goodbye!"
                )

                break

            # --------------------------------------------------
            # Clear conversation
            # --------------------------------------------------

            if user_input.lower() == "clear":

                conversation.clear()

                print()
                print(
                    "Conversation cleared."
                )

                print()

                continue

            # --------------------------------------------------
            # Build prompt
            # --------------------------------------------------

            conversation.append(
                (
                    "User",
                    user_input
                )
            )

            prompt_parts = []

            for role, message in conversation:

                prompt_parts.append(
                    f"{role}: {message}"
                )

            prompt_parts.append(
                "Geniee:"
            )

            prompt = "\n".join(
                prompt_parts
            )

            # --------------------------------------------------
            # Generate
            # --------------------------------------------------

            try:

                generated_text = (
                    self.respond(
                        prompt
                    )
                )

            except Exception as error:

                print()
                print(
                    "Generation error:"
                )

                print(
                    error
                )

                # Remove the user message because
                # generation failed.

                conversation.pop()

                print()

                continue

            # --------------------------------------------------
            # Extract response
            # --------------------------------------------------

            response = (
                self.extract_response(
                    generated_text
                )
            )

            print()
            print(
                f"Geniee: {response}"
            )

            print()

            conversation.append(
                (
                    "Geniee",
                    response
                )
            )

    # ======================================================
    # Extract Geniee response
    # ======================================================

    @staticmethod
    def extract_response(
        generated_text
    ):
        """
        Extract the Geniee portion from the generated text.

        Example:

            User: Hello
            Geniee: Hello! How can I help?

        becomes:

            Hello! How can I help?
        """

        if not generated_text:

            return ""

        text = generated_text.strip()

        if "Geniee:" in text:

            text = text.split(
                "Geniee:",
                1
            )[1]

        if "User:" in text:

            text = text.split(
                "User:",
                1
            )[0]

        return text.strip()


# ==========================================================
# Load chatbot
# ==========================================================

def load_chatbot():

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            "Geniee checkpoint not found:\n"
            f"{CHECKPOINT_PATH}\n\n"
            "Run training first."
        )

    print()
    print(
        "Loading Geniee..."
    )

    generator = load_geniee(
        CHECKPOINT_PATH
    )

    print(
        "Geniee loaded successfully."
    )

    return GenieeChat(
        generator
    )


# ==========================================================
# Main
# ==========================================================

def main():

    chatbot = load_chatbot()

    chatbot.start()


# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":

    main()