using UnityEngine;

namespace Vertical
{
    public sealed class MissionController : MonoBehaviour
    {
        private const string CompletedKey = "Vertical.ChapterOneComplete";

        private PlayerTraversal player;
        private GameUI ui;
        private bool active;
        private bool completed;

        public void Configure(PlayerTraversal traversal)
        {
            player = traversal;
            player.AltitudeChanged += OnAltitudeChanged;
            player.StateChanged += OnTraversalStateChanged;
        }

        public void SetUI(GameUI gameUi) => ui = gameUi;

        public void BeginMission()
        {
            completed = false;
            active = true;
            player.RespawnAtCheckpoint();
            ui.ShowGameplay();
            ui.ShowToast("SERVICE SPINE // Reach the amber service ledge");
        }

        public void ReachCheckpoint(Vector3 position)
        {
            if (!active || completed)
                return;

            player.SetCheckpoint(position);
            ui.ShowToast("CHECKPOINT SECURED");
        }

        public void FinishMission()
        {
            if (!active || completed)
                return;

            completed = true;
            active = false;
            PlayerPrefs.SetInt(CompletedKey, 1);
            PlayerPrefs.Save();
            player.Complete();
            ui.ShowComplete();
        }

        public void RestartMission()
        {
            Time.timeScale = 1f;
            completed = false;
            active = true;
            player.RespawnAtCheckpoint();
            ui.ShowGameplay();
            ui.ShowToast("ASCENT RESTARTED");
        }

        public void ReturnToTitle()
        {
            Time.timeScale = 1f;
            active = false;
            player.RespawnAtCheckpoint();
            ui.ShowTitle();
        }

        public bool HasCompletedChapter() => PlayerPrefs.GetInt(CompletedKey, 0) == 1;

        private void OnAltitudeChanged(float altitude)
        {
            if (active && ui != null)
                ui.SetAltitude(altitude);
        }

        private void OnTraversalStateChanged(TraversalState state)
        {
            if (active && ui != null)
                ui.SetTraversalState(state);
        }
    }

    public sealed class CheckpointLedge : MonoBehaviour
    {
        private void OnCollisionEnter(Collision collision)
        {
            var player = collision.collider.GetComponent<PlayerTraversal>();
            if (player == null)
                return;

            var mission = FindObjectOfType<MissionController>();
            if (mission != null)
                mission.ReachCheckpoint(transform.position);
        }
    }

    public sealed class ObjectiveLedge : MonoBehaviour
    {
        private MissionController mission;
        public void Configure(MissionController controller) => mission = controller;

        private void OnCollisionEnter(Collision collision)
        {
            if (collision.collider.GetComponent<PlayerTraversal>() != null)
                mission?.FinishMission();
        }
    }
}
