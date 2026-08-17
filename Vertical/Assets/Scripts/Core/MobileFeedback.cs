using UnityEngine;

namespace Vertical
{
    [RequireComponent(typeof(AudioSource))]
    public sealed class MobileFeedback : MonoBehaviour
    {
        private AudioSource source;
        private AudioClip grappleTone;
        private AudioClip glideTone;

        public void Configure(PlayerTraversal player)
        {
            source = GetComponent<AudioSource>();
            source.playOnAwake = false;
            source.spatialBlend = 0f;
            source.volume = 0.22f;
            grappleTone = CreateTone("Grapple Tone", 660f, 0.10f);
            glideTone = CreateTone("Glide Tone", 420f, 0.13f);
            player.StateChanged += OnStateChanged;
        }

        private void OnStateChanged(TraversalState state)
        {
            if (state == TraversalState.Swinging)
                Signal(grappleTone, 1.05f, true);
            else if (state == TraversalState.Gliding)
                Signal(glideTone, 0.85f, false);
            else if (state == TraversalState.Complete)
                Signal(grappleTone, 0.68f, true);
        }

        private void Signal(AudioClip clip, float pitch, bool haptic)
        {
            if (GameSettings.AudioEnabled && clip != null)
            {
                source.pitch = pitch;
                source.PlayOneShot(clip);
            }

            if (haptic && GameSettings.HapticsEnabled)
                Handheld.Vibrate();
        }

        private static AudioClip CreateTone(string name, float frequency, float duration)
        {
            const int sampleRate = 22050;
            var sampleCount = Mathf.CeilToInt(sampleRate * duration);
            var samples = new float[sampleCount];
            for (var i = 0; i < sampleCount; i++)
            {
                var fade = 1f - i / (float)sampleCount;
                samples[i] = Mathf.Sin(2f * Mathf.PI * frequency * i / sampleRate) * fade * 0.35f;
            }

            var clip = AudioClip.Create(name, sampleCount, 1, sampleRate, false);
            clip.SetData(samples, 0);
            return clip;
        }
    }
}
