using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace Vertical
{
    public sealed class GameUI : MonoBehaviour
    {
        private static readonly Color Panel = new Color(0.027f, 0.039f, 0.059f, 0.94f);
        private static readonly Color Card = new Color(0.055f, 0.090f, 0.115f, 0.94f);
        private static readonly Color Ink = new Color(0.95f, 0.97f, 0.98f);

        private Canvas canvas;
        private PlayerTraversal player;
        private MissionController mission;
        private GameObject titleRoot;
        private GameObject briefingRoot;
        private GameObject pauseRoot;
        private GameObject completeRoot;
        private GameObject settingsRoot;
        private GameObject hudRoot;
        private Text altitudeText;
        private Text stateText;
        private Text qualityText;
        private Text toastText;
        private Image reticle;
        private float toastUntil;

        public bool InterceptsGameplay => titleRoot.activeSelf || briefingRoot.activeSelf || pauseRoot.activeSelf || completeRoot.activeSelf || settingsRoot.activeSelf;

        public static GameUI Create(PlayerTraversal traversal, MissionController controller)
        {
            var root = new GameObject("Game UI", typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
            var ui = root.AddComponent<GameUI>();
            ui.Initialise(traversal, controller);
            return ui;
        }

        private void Initialise(PlayerTraversal traversal, MissionController controller)
        {
            player = traversal;
            mission = controller;
            canvas = GetComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 20;
            var scaler = GetComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1080f, 1920f);
            scaler.matchWidthOrHeight = 0.5f;

            if (FindObjectOfType<EventSystem>() == null)
            {
                var eventSystem = new GameObject("EventSystem", typeof(EventSystem), typeof(StandaloneInputModule));
                eventSystem.transform.SetParent(transform, false);
            }

            BuildHUD();
            titleRoot = BuildTitle();
            briefingRoot = BuildBriefing();
            pauseRoot = BuildPause();
            completeRoot = BuildComplete();
            settingsRoot = BuildSettings();
            ShowTitle();
        }

        private void Update()
        {
            if (toastText != null && toastText.gameObject.activeSelf && Time.unscaledTime > toastUntil)
                toastText.gameObject.SetActive(false);
        }

        public void ShowTitle()
        {
            Time.timeScale = 1f;
            titleRoot.SetActive(true);
            briefingRoot.SetActive(false);
            pauseRoot.SetActive(false);
            completeRoot.SetActive(false);
            settingsRoot.SetActive(false);
            hudRoot.SetActive(false);
        }

        public void ShowBriefing()
        {
            titleRoot.SetActive(false);
            briefingRoot.SetActive(true);
            hudRoot.SetActive(false);
        }

        public void ShowGameplay()
        {
            titleRoot.SetActive(false);
            briefingRoot.SetActive(false);
            pauseRoot.SetActive(false);
            completeRoot.SetActive(false);
            settingsRoot.SetActive(false);
            hudRoot.SetActive(true);
            UpdateQualityLabel();
        }

        public void ShowComplete()
        {
            completeRoot.SetActive(true);
            hudRoot.SetActive(false);
        }

        private void ShowSettings()
        {
            titleRoot.SetActive(false);
            settingsRoot.SetActive(true);
            RefreshSettingsLabels();
        }

        private void CloseSettings()
        {
            settingsRoot.SetActive(false);
            titleRoot.SetActive(true);
            UpdateQualityLabel();
        }

        public void TogglePause()
        {
            if (pauseRoot.activeSelf)
            {
                pauseRoot.SetActive(false);
                Time.timeScale = 1f;
            }
            else
            {
                pauseRoot.SetActive(true);
                Time.timeScale = 0f;
            }
        }

        public void SetAltitude(float altitude)
        {
            altitudeText.text = $"ALT {altitude:000}m";
        }

        public void SetTraversalState(TraversalState state)
        {
            var prompt = state == TraversalState.Swinging ? "RELEASE TO LAUNCH" : state == TraversalState.Gliding ? "GLIDE // HOLD" : "TAP CYAN TARGET";
            stateText.text = prompt;
        }

        public void UpdateReticle(Vector2 screenPosition, bool valid)
        {
            reticle.gameObject.SetActive(valid);
            if (!valid)
                return;

            var point = new Vector2(screenPosition.x - Screen.width * 0.5f, screenPosition.y - Screen.height * 0.5f);
            reticle.rectTransform.anchoredPosition = point;
            reticle.color = valid ? VerticalBootstrap.Cyan : Color.white;
        }

        public void ShowToast(string message)
        {
            toastText.text = message;
            toastText.gameObject.SetActive(true);
            toastUntil = Time.unscaledTime + 2.2f;
        }

        private GameObject BuildHUD()
        {
            var root = CreatePanel("HUD", transform, Color.clear);
            hudRoot = root;
            var rect = root.GetComponent<RectTransform>();
            Stretch(rect);
            root.GetComponent<Image>().raycastTarget = false;

            var chapter = CreateText("Chapter", root.transform, "CHAPTER 01 // SERVICE SPINE", 29, Ink, TextAnchor.UpperLeft);
            Anchor(chapter.rectTransform, new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(54f, -62f), new Vector2(560f, 54f));
            altitudeText = CreateText("Altitude", root.transform, "ALT 000m", 29, Ink, TextAnchor.UpperRight);
            Anchor(altitudeText.rectTransform, new Vector2(1f, 1f), new Vector2(1f, 1f), new Vector2(-382f, -62f), new Vector2(330f, 54f));

            var pause = CreateButton("Pause", root.transform, "Ⅱ", 36, new Color(0.1f, 0.15f, 0.18f, 0.95f), TogglePause);
            Anchor(pause.GetComponent<RectTransform>(), new Vector2(1f, 1f), new Vector2(1f, 1f), new Vector2(-112f, -76f), new Vector2(72f, 72f));
            stateText = CreateText("State", root.transform, "TAP CYAN TARGET", 28, VerticalBootstrap.Cyan, TextAnchor.MiddleCenter);
            Anchor(stateText.rectTransform, new Vector2(0.5f, 0f), new Vector2(0.5f, 0f), new Vector2(-300f, 185f), new Vector2(600f, 52f));

            toastText = CreateText("Toast", root.transform, string.Empty, 25, VerticalBootstrap.Amber, TextAnchor.MiddleCenter);
            Anchor(toastText.rectTransform, new Vector2(0.5f, 0.72f), new Vector2(0.5f, 0.72f), new Vector2(-410f, -30f), new Vector2(820f, 60f));
            toastText.gameObject.SetActive(false);

            reticle = CreateImage("Target Reticle", root.transform, VerticalBootstrap.Cyan);
            reticle.rectTransform.sizeDelta = new Vector2(46f, 46f);
            reticle.rectTransform.anchorMin = reticle.rectTransform.anchorMax = new Vector2(0.5f, 0.5f);
            reticle.raycastTarget = false;
            reticle.gameObject.SetActive(false);
            return root;
        }

        private GameObject BuildTitle()
        {
            var root = CreateFullScreenPanel("Title", Panel);
            var signal = CreateText("Signal", root.transform, "// STORM PROTOCOL", 28, VerticalBootstrap.Cyan, TextAnchor.MiddleCenter);
            Anchor(signal.rectTransform, new Vector2(0.5f, 0.75f), new Vector2(0.5f, 0.75f), new Vector2(-360f, -30f), new Vector2(720f, 52f));
            var title = CreateText("Title", root.transform, "VERTICAL", 114, Ink, TextAnchor.MiddleCenter);
            title.fontStyle = FontStyle.Bold;
            Anchor(title.rectTransform, new Vector2(0.5f, 0.64f), new Vector2(0.5f, 0.64f), new Vector2(-480f, -85f), new Vector2(960f, 150f));
            var description = CreateText("Description", root.transform, "THE TOWER REMEMBERS WHAT THEY BURIED.", 25, new Color(0.70f, 0.78f, 0.81f), TextAnchor.MiddleCenter);
            Anchor(description.rectTransform, new Vector2(0.5f, 0.54f), new Vector2(0.5f, 0.54f), new Vector2(-450f, -45f), new Vector2(900f, 80f));

            var start = CreateButton("Start", root.transform, mission.HasCompletedChapter() ? "CONTINUE ASCENT" : "START ASCENT", 31, VerticalBootstrap.Cyan, ShowBriefing);
            Anchor(start.GetComponent<RectTransform>(), new Vector2(0.5f, 0.35f), new Vector2(0.5f, 0.35f), new Vector2(-330f, -62f), new Vector2(660f, 100f));
            var quality = CreateButton("Quality", root.transform, "GRAPHICS", 25, Card, CycleQuality);
            Anchor(quality.GetComponent<RectTransform>(), new Vector2(0.5f, 0.27f), new Vector2(0.5f, 0.27f), new Vector2(-330f, -50f), new Vector2(660f, 82f));
            qualityText = quality.GetComponentInChildren<Text>();
            UpdateQualityLabel();

            var settings = CreateButton("Settings", root.transform, "SETTINGS", 25, Card, ShowSettings);
            Anchor(settings.GetComponent<RectTransform>(), new Vector2(0.5f, 0.19f), new Vector2(0.5f, 0.19f), new Vector2(-330f, -50f), new Vector2(660f, 82f));
            var note = CreateText("Instruction", root.transform, "Tap cyan targets. Release to launch. Hold in air to glide.", 22, new Color(0.60f, 0.70f, 0.74f), TextAnchor.MiddleCenter);
            Anchor(note.rectTransform, new Vector2(0.5f, 0.09f), new Vector2(0.5f, 0.09f), new Vector2(-440f, -55f), new Vector2(880f, 90f));
            return root;
        }

        private GameObject BuildSettings()
        {
            var root = CreateFullScreenPanel("Settings", Panel);
            var header = CreateText("Header", root.transform, "FIELD SETTINGS", 50, Ink, TextAnchor.MiddleCenter);
            Anchor(header.rectTransform, new Vector2(0.5f, 0.74f), new Vector2(0.5f, 0.74f), new Vector2(-420f, -65f), new Vector2(840f, 110f));
            var haptics = CreateButton("Haptics", root.transform, "HAPTICS", 28, Card, ToggleHaptics);
            haptics.gameObject.AddComponent<SettingLabel>().Configure(SettingLabelType.Haptics);
            Anchor(haptics.GetComponent<RectTransform>(), new Vector2(0.5f, 0.56f), new Vector2(0.5f, 0.56f), new Vector2(-330f, -50f), new Vector2(660f, 90f));
            var audio = CreateButton("Audio", root.transform, "AUDIO", 28, Card, ToggleAudio);
            audio.gameObject.AddComponent<SettingLabel>().Configure(SettingLabelType.Audio);
            Anchor(audio.GetComponent<RectTransform>(), new Vector2(0.5f, 0.46f), new Vector2(0.5f, 0.46f), new Vector2(-330f, -50f), new Vector2(660f, 90f));
            var sensitivity = CreateButton("Sensitivity", root.transform, "SWIPE SENSITIVITY", 28, Card, CycleSensitivity);
            sensitivity.gameObject.AddComponent<SettingLabel>().Configure(SettingLabelType.Sensitivity);
            Anchor(sensitivity.GetComponent<RectTransform>(), new Vector2(0.5f, 0.36f), new Vector2(0.5f, 0.36f), new Vector2(-330f, -50f), new Vector2(660f, 90f));
            var back = CreateButton("Back", root.transform, "BACK", 28, VerticalBootstrap.Cyan, CloseSettings);
            Anchor(back.GetComponent<RectTransform>(), new Vector2(0.5f, 0.20f), new Vector2(0.5f, 0.20f), new Vector2(-330f, -50f), new Vector2(660f, 90f));
            return root;
        }

        private GameObject BuildBriefing()
        {
            var root = CreateFullScreenPanel("Briefing", Panel);
            var chapter = CreateText("Header", root.transform, "CHAPTER 01 // THE SERVICE SPINE", 32, VerticalBootstrap.Cyan, TextAnchor.MiddleCenter);
            Anchor(chapter.rectTransform, new Vector2(0.5f, 0.72f), new Vector2(0.5f, 0.72f), new Vector2(-460f, -46f), new Vector2(920f, 80f));
            var body = CreateText("Briefing", root.transform, "Two years after the collapse, the tower still carries the scars. Find the amber maintenance ledge and recover the first authorization trace.", 33, Ink, TextAnchor.MiddleCenter);
            body.horizontalOverflow = HorizontalWrapMode.Wrap;
            Anchor(body.rectTransform, new Vector2(0.5f, 0.52f), new Vector2(0.5f, 0.52f), new Vector2(-400f, -150f), new Vector2(800f, 290f));
            var start = CreateButton("Begin", root.transform, "BEGIN", 31, VerticalBootstrap.Cyan, mission.BeginMission);
            Anchor(start.GetComponent<RectTransform>(), new Vector2(0.5f, 0.26f), new Vector2(0.5f, 0.26f), new Vector2(-330f, -62f), new Vector2(660f, 100f));
            return root;
        }

        private GameObject BuildPause()
        {
            var root = CreateFullScreenPanel("Pause", Panel);
            var header = CreateText("Header", root.transform, "ASCENT PAUSED", 52, Ink, TextAnchor.MiddleCenter);
            Anchor(header.rectTransform, new Vector2(0.5f, 0.68f), new Vector2(0.5f, 0.68f), new Vector2(-400f, -70f), new Vector2(800f, 120f));
            var resume = CreateButton("Resume", root.transform, "RESUME", 29, VerticalBootstrap.Cyan, TogglePause);
            Anchor(resume.GetComponent<RectTransform>(), new Vector2(0.5f, 0.48f), new Vector2(0.5f, 0.48f), new Vector2(-330f, -55f), new Vector2(660f, 90f));
            var restart = CreateButton("Restart", root.transform, "RESTART CHECKPOINT", 27, Card, mission.RestartMission);
            Anchor(restart.GetComponent<RectTransform>(), new Vector2(0.5f, 0.39f), new Vector2(0.5f, 0.39f), new Vector2(-330f, -55f), new Vector2(660f, 90f));
            var title = CreateButton("Title", root.transform, "RETURN TO TITLE", 27, Card, mission.ReturnToTitle);
            Anchor(title.GetComponent<RectTransform>(), new Vector2(0.5f, 0.30f), new Vector2(0.5f, 0.30f), new Vector2(-330f, -55f), new Vector2(660f, 90f));
            return root;
        }

        private GameObject BuildComplete()
        {
            var root = CreateFullScreenPanel("Complete", Panel);
            var header = CreateText("Header", root.transform, "ACCESS TRACE RECOVERED", 46, VerticalBootstrap.Amber, TextAnchor.MiddleCenter);
            Anchor(header.rectTransform, new Vector2(0.5f, 0.70f), new Vector2(0.5f, 0.70f), new Vector2(-460f, -70f), new Vector2(920f, 120f));
            var body = CreateText("Body", root.transform, "The signature is real. Someone approved the material substitution, then sealed the record behind the executive floors.", 31, Ink, TextAnchor.MiddleCenter);
            Anchor(body.rectTransform, new Vector2(0.5f, 0.51f), new Vector2(0.5f, 0.51f), new Vector2(-410f, -145f), new Vector2(820f, 280f));
            var replay = CreateButton("Replay", root.transform, "REPLAY CHAPTER", 29, VerticalBootstrap.Cyan, mission.RestartMission);
            Anchor(replay.GetComponent<RectTransform>(), new Vector2(0.5f, 0.30f), new Vector2(0.5f, 0.30f), new Vector2(-330f, -55f), new Vector2(660f, 90f));
            var title = CreateButton("Return", root.transform, "RETURN TO TITLE", 27, Card, mission.ReturnToTitle);
            Anchor(title.GetComponent<RectTransform>(), new Vector2(0.5f, 0.22f), new Vector2(0.5f, 0.22f), new Vector2(-330f, -55f), new Vector2(660f, 90f));
            return root;
        }

        private void CycleQuality()
        {
            QualityController.CycleProfile();
            UpdateQualityLabel();
            ShowToast($"GRAPHICS // {QualityController.Current.ToString().ToUpperInvariant()}");
        }

        private void ToggleHaptics()
        {
            GameSettings.ToggleHaptics();
            RefreshSettingsLabels();
        }

        private void ToggleAudio()
        {
            GameSettings.ToggleAudio();
            RefreshSettingsLabels();
        }

        private void CycleSensitivity()
        {
            GameSettings.CycleSensitivity();
            RefreshSettingsLabels();
        }

        private void RefreshSettingsLabels()
        {
            foreach (var label in settingsRoot.GetComponentsInChildren<SettingLabel>())
                label.Refresh();
        }

        private void UpdateQualityLabel()
        {
            if (qualityText != null)
                qualityText.text = $"GRAPHICS // {QualityController.Current.ToString().ToUpperInvariant()}";
        }

        private GameObject CreateFullScreenPanel(string name, Color color)
        {
            var panel = CreatePanel(name, transform, color);
            Stretch(panel.GetComponent<RectTransform>());
            return panel;
        }

        private static GameObject CreatePanel(string name, Transform parent, Color color)
        {
            var panel = new GameObject(name, typeof(RectTransform), typeof(Image));
            panel.transform.SetParent(parent, false);
            panel.GetComponent<Image>().color = color;
            return panel;
        }

        private static Image CreateImage(string name, Transform parent, Color color)
        {
            var image = new GameObject(name, typeof(RectTransform), typeof(Image)).GetComponent<Image>();
            image.transform.SetParent(parent, false);
            image.color = color;
            return image;
        }

        private static Text CreateText(string name, Transform parent, string content, int fontSize, Color color, TextAnchor alignment)
        {
            var text = new GameObject(name, typeof(RectTransform), typeof(Text)).GetComponent<Text>();
            text.transform.SetParent(parent, false);
            text.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
            text.text = content;
            text.fontSize = fontSize;
            text.color = color;
            text.alignment = alignment;
            text.horizontalOverflow = HorizontalWrapMode.Wrap;
            text.verticalOverflow = VerticalWrapMode.Overflow;
            return text;
        }

        private static Button CreateButton(string name, Transform parent, string label, int size, Color color, UnityEngine.Events.UnityAction action)
        {
            var buttonObject = CreatePanel(name, parent, color);
            var button = buttonObject.AddComponent<Button>();
            var colors = button.colors;
            colors.normalColor = Color.white;
            colors.highlightedColor = new Color(0.9f, 0.98f, 1f);
            colors.pressedColor = new Color(0.72f, 0.88f, 0.92f);
            button.colors = colors;
            button.onClick.AddListener(action);
            var labelColor = color == VerticalBootstrap.Cyan ? VerticalBootstrap.Void : Ink;
            var labelText = CreateText("Label", buttonObject.transform, label, size, labelColor, TextAnchor.MiddleCenter);
            Stretch(labelText.rectTransform);
            return button;
        }

        private static void Stretch(RectTransform rect)
        {
            rect.anchorMin = Vector2.zero;
            rect.anchorMax = Vector2.one;
            rect.offsetMin = Vector2.zero;
            rect.offsetMax = Vector2.zero;
        }

        private static void Anchor(RectTransform rect, Vector2 min, Vector2 max, Vector2 position, Vector2 size)
        {
            rect.anchorMin = min;
            rect.anchorMax = max;
            rect.anchoredPosition = position;
            rect.sizeDelta = size;
        }
    }

    public enum SettingLabelType { Haptics, Audio, Sensitivity }

    public sealed class SettingLabel : MonoBehaviour
    {
        private SettingLabelType type;

        public void Configure(SettingLabelType settingType)
        {
            type = settingType;
            Refresh();
        }

        public void Refresh()
        {
            var text = GetComponentInChildren<Text>();
            if (text == null)
                return;

            text.text = type == SettingLabelType.Haptics
                ? $"HAPTICS // {(GameSettings.HapticsEnabled ? "ON" : "OFF")}" 
                : type == SettingLabelType.Audio
                    ? $"AUDIO // {(GameSettings.AudioEnabled ? "ON" : "OFF")}" 
                    : $"SWIPE SENSITIVITY // {GameSettings.Sensitivity:0.0}x";
        }
    }
}
