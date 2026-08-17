using UnityEngine;

namespace Vertical
{
    public enum GraphicsProfile { Performance, Balanced, Cinematic }

    public static class QualityController
    {
        private const string ProfileKey = "Vertical.GraphicsProfile";
        public static GraphicsProfile Current { get; private set; }

        public static void ApplyStoredProfile()
        {
            Apply((GraphicsProfile)PlayerPrefs.GetInt(ProfileKey, (int)GraphicsProfile.Balanced));
        }

        public static void CycleProfile()
        {
            var next = (GraphicsProfile)(((int)Current + 1) % 3);
            Apply(next);
            PlayerPrefs.SetInt(ProfileKey, (int)next);
            PlayerPrefs.Save();
        }

        public static void Apply(GraphicsProfile profile)
        {
            Current = profile;
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = 30;
            QualitySettings.shadowDistance = profile == GraphicsProfile.Cinematic ? 32f : profile == GraphicsProfile.Balanced ? 20f : 12f;
            QualitySettings.shadowResolution = profile == GraphicsProfile.Cinematic ? ShadowResolution.High : ShadowResolution.Medium;
            QualitySettings.antiAliasing = profile == GraphicsProfile.Cinematic ? 4 : 2;
            QualitySettings.lodBias = profile == GraphicsProfile.Cinematic ? 1f : 0.65f;
            RenderSettings.fogDensity = profile == GraphicsProfile.Performance ? 0.012f : profile == GraphicsProfile.Balanced ? 0.018f : 0.023f;
        }
    }
}
