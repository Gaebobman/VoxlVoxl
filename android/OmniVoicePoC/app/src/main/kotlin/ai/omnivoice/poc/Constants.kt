package ai.omnivoice.poc

/**
 * Constants lifted from k2-fsa/OmniVoice `config.json` and the Higgs Audio V2
 * tokenizer config. Mirrored in `models/android/manifest.json`; see
 * `docs/model-analysis.md` §1.
 */
object OV {
    const val NUM_CODEBOOKS = 8
    const val AUDIO_VOCAB_SIZE = 1025
    const val AUDIO_MASK_ID = 1024
    const val HIDDEN_SIZE = 1024

    const val SR_24K = 24_000
    const val SR_16K = 16_000
    const val HOP_LENGTH = 960
    const val FRAME_RATE = SR_24K / HOP_LENGTH   // 25 fps

    val CODEBOOK_WEIGHTS = intArrayOf(8, 8, 6, 6, 4, 4, 2, 2)

    const val TOK_DENOISE = "<|denoise|>"
    const val TOK_LANG_START = "<|lang_start|>"
    const val TOK_LANG_END = "<|lang_end|>"
    const val TOK_INSTRUCT_START = "<|instruct_start|>"
    const val TOK_INSTRUCT_END = "<|instruct_end|>"
    const val TOK_TEXT_START = "<|text_start|>"
    const val TOK_TEXT_END = "<|text_end|>"
}

/**
 * Upstream `OmniVoiceGenerationConfig`. [guidanceScale] must not be zero:
 * without the unconditional branch the model emits pure silence
 * (`docs/benchmark.md` §2), so it is validated rather than merely documented.
 */
data class GenConfig(
    val numStep: Int = 16,
    val guidanceScale: Float = 2.0f,
    /** float64 on purpose: a Float round-trip shifts the schedule. */
    val tShift: Double = 0.1,
    val layerPenaltyFactor: Float = 5.0f,
    val positionTemperature: Float = 5.0f,
    val classTemperature: Float = 0.0f,
    val denoise: Boolean = true,
    val postprocessOutput: Boolean = true,
    val padSeconds: Float = 0.1f,
    val fadeSeconds: Float = 0.1f,
) {
    init {
        require(numStep in 1..128) { "numStep out of range: $numStep" }
        require(guidanceScale != 0.0f) {
            "guidanceScale=0 disables classifier-free guidance, which makes OmniVoice " +
                "emit silence rather than speech. See docs/benchmark.md §2."
        }
    }
}

/** Every failure the PoC is expected to surface, with a stable tag for Logcat. */
class OmniVoiceException(
    val kind: Kind,
    message: String,
    cause: Throwable? = null,
) : Exception("[${kind.name}] $message", cause) {
    enum class Kind {
        MODEL_MISSING, MODEL_LOAD_FAILED, UNSUPPORTED_OP, WAV_DECODE_FAILED,
        BAD_SAMPLE_RATE, TOKENIZER_FAILED, OUT_OF_MEMORY, GENERATION_FAILED,
        VOICE_PROMPT_INVALID,
    }
}
