using UnityEngine;

namespace Vertical
{
    public static class GameSettings
    {
        private const string HapticsKey = "Vertical.Haptics";
        private const string AudioKey = "Vertical.Audio";
        private const string SensitivityKey = "Vertical.Sensitivity";

        public static bool HapticsEnabled => PlayerPrefs.GetInt(HapticsKey, 1) == 1;
        public static bool AudioEnabled => PlayerPrefs.GetInt(AudioKey, 1) == 1;
        public static float Sensitivity => PlayerPrefs.GetFloat(SensitivityKey, 1f);

        public static void ToggleHaptics() => SetFlag(HapticsKey, !HapticsEnabled);
        public static void ToggleAudio() => SetFlag(AudioKey, !AudioEnabled);

        public static void CycleSensitivity()
        {
            var next = Sensitivity >= 1.25f ? 0.8f : Sensitivity >= 1f ? 1.3f : 1f;
            PlayerPrefs.SetFloat(SensitivityKey, next);
            PlayerPrefs.Save();
        }

        private static void SetFlag(string key, bool value)
        {
            PlayerPrefs.SetInt(key, value ? 1 : 0);
            PlayerPrefs.Save();
        }
    }
}
