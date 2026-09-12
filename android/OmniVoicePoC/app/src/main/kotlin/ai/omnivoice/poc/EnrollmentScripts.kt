package ai.omnivoice.poc

/**
 * Sentences the user is asked to READ during enrollment.
 *
 * The transcript is the silent failure mode of this whole pipeline: it is
 * concatenated ahead of the target text and is what aligns the reference codes
 * with the reference speech, so a wrong one degrades the voice with no error
 * anywhere. Handing the user a sentence removes the error class instead of
 * asking them to avoid it.
 *
 * Each script is chosen to be long enough to clear the 3 s floor and short
 * enough to stay under ~12 s, since a longer reference slows every later
 * generation (it is re-read at all 32 forwards) without improving the voice.
 */
object EnrollmentScripts {

    data class Script(val language: String, val label: String, val text: String)

    val ALL = listOf(
        Script("ko", "기본", "이것은 제 목소리를 등록하기 위한 짧은 문장입니다. 편한 속도로 읽어 주세요."),
        Script("ko", "긴 문장", "안녕하세요. 오늘은 날씨가 참 좋네요. 잠시 뒤에 다시 연락드리겠습니다. 감사합니다."),
        Script("ko", "차분하게", "조금 천천히, 낮은 목소리로 읽어 주세요. 지금 이 말투가 그대로 남습니다."),
        Script("ko", "속삭임", "아주 작은 목소리로 속삭이듯 읽어 주세요. 목소리를 내지 않아도 괜찮습니다."),
        Script("en", "English", "This is a short sentence for registering my voice. Please read it at a comfortable pace."),
    )

    fun forLanguage(code: String?): List<Script> =
        if (code == null) ALL else ALL.filter { it.language == code }.ifEmpty { ALL }

    /** The sentence a freshly enrolled voice is tested with (spec §6). */
    const val TEST_SENTENCE = "안녕하세요. VoxlVoxl에 등록된 제 목소리입니다."
}
