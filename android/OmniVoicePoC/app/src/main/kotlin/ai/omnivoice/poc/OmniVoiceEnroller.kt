package ai.omnivoice.poc

import ai.omnivoice.poc.core.VoiceEnroller
import ai.omnivoice.poc.core.VoiceProfile
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Spec F-A01 / F-A02 — reference audio to [VoiceProfile], on device.
 *
 * The three encoder graphs total 654 MB, against 509 MB for the generation path.
 * They are needed **once per speaker** and never during synthesis, so this class
 * opens them, uses them and closes them inside one `use {}` block: the two sets
 * of weights must not be resident at the same time.
 *
 *   waveform 24 kHz -> acoustic_encoder  -> [1,256,T]
 *   waveform 16 kHz -> semantic_encoder  -> [1,768,T]
 *                      quantizer_encoder -> codes [8,1,T]
 *
 * Mirrors `create_voice_clone_prompt` and `scripts/infer_onnx.py enroll`, whose
 * output this reproduces to 92 % code agreement against PyTorch (99 % on
 * codebook 0, which carries most of the speaker identity).
 */
class OmniVoiceEnroller(
    private val modelDir: File,
    private val threads: Int = 0,
) : VoiceEnroller {

    companion object {
        const val TAG = "OmniVoice.Enroll"
        val REQUIRED = listOf(
            "acoustic_encoder.onnx", "semantic_encoder.onnx", "quantizer_encoder.onnx")

        /** True when this build can enroll rather than only play back. */
        fun available(modelDir: File) = REQUIRED.all { File(modelDir, it).isFile }
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val sessions = HashMap<String, OrtSession>()

    var lastEncodeMillis: Long = 0; private set

    private fun session(name: String): OrtSession = sessions.getOrPut(name) {
        val f = File(modelDir, name)
        if (!f.isFile) {
            throw OmniVoiceException(
                OmniVoiceException.Kind.MODEL_MISSING,
                "$name not found in $modelDir — this build cannot enroll; " +
                    "push the encoder graphs or import a profile made on PC",
            )
        }
        val so = OrtSession.SessionOptions()
        so.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        if (threads > 0) so.setIntraOpNumThreads(threads)
        try {
            env.createSession(f.absolutePath, so)
        } catch (e: OrtException) {
            throw OmniVoiceException(
                OmniVoiceException.Kind.MODEL_LOAD_FAILED, "failed to load $name", e)
        } catch (e: OutOfMemoryError) {
            throw OmniVoiceException(
                OmniVoiceException.Kind.OUT_OF_MEMORY,
                "OOM loading $name — the encoders need ~654 MB on top of anything " +
                    "already open; close the synthesizer first",
            )
        }
    }

    override fun enroll(
        samples: FloatArray,
        sampleRate: Int,
        refText: String,
        displayName: String,
    ): VoiceProfile {
        if (refText.isBlank()) {
            throw OmniVoiceException(
                OmniVoiceException.Kind.TOKENIZER_FAILED,
                "reference transcript is required — it is concatenated ahead of the " +
                    "target text and is what aligns the reference codes with the speech",
            )
        }

        var x24 = if (sampleRate == OV.SR_24K) samples
        else WavIo.resample(samples, sampleRate, OV.SR_24K)

        if (x24.size < OV.SR_24K) {                       // < 1 s
            throw OmniVoiceException(
                OmniVoiceException.Kind.BAD_SAMPLE_RATE,
                "reference is ${"%.2f".format(x24.size.toDouble() / OV.SR_24K)}s; " +
                    "3-10 s is the useful range",
            )
        }

        // RMS is captured BEFORE trimming and is replayed onto the output, so a
        // quiet enrollment produces quiet synthesis rather than a level jump.
        val refRms = SilenceUtils.rms(x24)
        if (refRms > 0f && refRms < 0.1f) {
            val g = 0.1f / refRms
            x24 = FloatArray(x24.size) { x24[it] * g }
        }
        x24 = SilenceUtils.removeSilence(x24, OV.SR_24K, 200, 100, 200)
        if (x24.isEmpty()) {
            throw OmniVoiceException(
                OmniVoiceException.Kind.WAV_DECODE_FAILED,
                "reference is entirely below the -50 dBFS silence floor — nothing was recorded",
            )
        }
        x24 = SilenceUtils.clipToHop(x24)
        val x16 = WavIo.resample(x24, OV.SR_24K, OV.SR_16K)

        val t0 = System.nanoTime()
        val acoustic = runEncoder("acoustic_encoder.onnx", "waveform_24k",
            longArrayOf(1, 1, x24.size.toLong()), x24, "acoustic_features")
        val semantic = runEncoder("semantic_encoder.onnx", "waveform_16k",
            longArrayOf(1, x16.size.toLong()), x16, "semantic_features")

        // acoustic and semantic can differ by one frame on non-multiple lengths
        val frames = minOf(acoustic.frames, semantic.frames)
        val codes = runQuantizer(acoustic, semantic, frames)
        lastEncodeMillis = (System.nanoTime() - t0) / 1_000_000

        Log.i(TAG, "enrolled '$displayName' in ${lastEncodeMillis}ms: " +
            "${codes[0].size} frames (${"%.2f".format(codes[0].size.toFloat() / OV.FRAME_RATE)}s), " +
            "rms=$refRms")

        return VoiceProfile(
            id = "%s_%d".format(displayName.replace(Regex("[^A-Za-z0-9_-]"), "_").take(32),
                System.currentTimeMillis() % 100000),
            displayName = displayName,
            codes = codes,
            refText = TextUtils.addPunctuation(refText),
            refRms = refRms,
            sampleRate = OV.SR_24K,
        )
    }

    private class Features(val data: FloatArray, val channels: Int, val frames: Int)

    private fun runEncoder(
        model: String, inputName: String, shape: LongArray,
        wav: FloatArray, outputName: String,
    ): Features {
        val buf = ByteBuffer.allocateDirect(wav.size * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
        buf.put(wav); buf.rewind()
        val t = OnnxTensor.createTensor(env, buf, shape)
        try {
            session(model).run(mapOf(inputName to t)).use { out ->
                val tensor = out[0] as OnnxTensor
                val dims = tensor.info.shape           // [1, C, T]
                val fb = tensor.floatBuffer
                val arr = FloatArray(fb.remaining())
                fb.get(arr)
                return Features(arr, dims[1].toInt(), dims[2].toInt())
            }
        } catch (e: OrtException) {
            throw OmniVoiceException(
                OmniVoiceException.Kind.GENERATION_FAILED, "$model failed", e)
        } finally {
            t.close()
        }
    }

    private fun runQuantizer(a: Features, s: Features, frames: Int): Array<ShortArray> {
        fun trim(f: Features): ByteBuffer {
            val b = ByteBuffer.allocateDirect(f.channels * frames * 4).order(ByteOrder.nativeOrder())
            val fb = b.asFloatBuffer()
            for (c in 0 until f.channels) {
                fb.put(f.data, c * f.frames, frames)
            }
            return b
        }

        val ta = OnnxTensor.createTensor(env, trim(a).asFloatBuffer(),
            longArrayOf(1, a.channels.toLong(), frames.toLong()))
        val ts = OnnxTensor.createTensor(env, trim(s).asFloatBuffer(),
            longArrayOf(1, s.channels.toLong(), frames.toLong()))
        try {
            session("quantizer_encoder.onnx").run(
                mapOf("acoustic_features" to ta, "semantic_features" to ts)
            ).use { out ->
                val lb = (out[0] as OnnxTensor).longBuffer      // [8, 1, T]
                val codes = Array(OV.NUM_CODEBOOKS) { ShortArray(frames) }
                for (c in 0 until OV.NUM_CODEBOOKS) {
                    for (t in 0 until frames) codes[c][t] = lb.get().toShort()
                }
                return codes
            }
        } catch (e: OrtException) {
            throw OmniVoiceException(
                OmniVoiceException.Kind.GENERATION_FAILED, "quantizer_encoder failed", e)
        } finally {
            ta.close(); ts.close()
        }
    }

    override fun close() {
        sessions.values.forEach { runCatching { it.close() } }
        sessions.clear()
    }
}
