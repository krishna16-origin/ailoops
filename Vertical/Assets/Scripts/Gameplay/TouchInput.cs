using UnityEngine;

namespace Vertical
{
    public sealed class TouchInput : MonoBehaviour
    {
        private const float TapWindow = 0.24f;
        private const float HoldWindow = 0.18f;
        private const float DoubleTapWindow = 0.32f;

        private PlayerTraversal player;
        private GameUI ui;
        private bool pressed;
        private bool holding;
        private float downTime;
        private float lastTapTime = -10f;
        private Vector2 downPosition;
        private Vector2 pointerPosition;

        public void Configure(PlayerTraversal traversal, GameUI gameUi)
        {
            player = traversal;
            ui = gameUi;
        }

        private void Update()
        {
            if (player == null || ui == null || ui.InterceptsGameplay)
                return;

            if (Input.touchCount > 0)
                HandleTouch(Input.GetTouch(0));
            else
                HandleMouse();

            if (pressed)
            {
                player.UpdateAim(pointerPosition, ui);
                var steerAmount = (pointerPosition.x - downPosition.x) / (Screen.width * 0.20f);
                player.SetSteering(steerAmount * GameSettings.Sensitivity);
                if (!player.IsSwinging && Time.time - downTime >= HoldWindow)
                {
                    holding = true;
                    player.SetGlide(true);
                }
            }
        }

        private void HandleTouch(Touch touch)
        {
            pointerPosition = touch.position;
            if (touch.phase == TouchPhase.Began)
                BeginPress(touch.position);
            else if (touch.phase == TouchPhase.Ended || touch.phase == TouchPhase.Canceled)
                EndPress(touch.position);
        }

        private void HandleMouse()
        {
            pointerPosition = Input.mousePosition;
            if (Input.GetMouseButtonDown(0))
                BeginPress(pointerPosition);
            if (Input.GetMouseButtonUp(0))
                EndPress(pointerPosition);
        }

        private void BeginPress(Vector2 position)
        {
            pressed = true;
            holding = false;
            downTime = Time.time;
            downPosition = position;
            player.UpdateAim(position, ui);
        }

        private void EndPress(Vector2 position)
        {
            if (!pressed)
                return;

            var duration = Time.time - downTime;
            var drag = position - downPosition;
            pressed = false;
            player.SetGlide(false);
            player.SetSteering(0f);

            if (player.IsSwinging)
            {
                player.ReleaseGrapple(drag);
                return;
            }

            if (holding)
                return;

            if (duration <= TapWindow)
            {
                if (Time.time - lastTapTime <= DoubleTapWindow)
                {
                    player.EmergencyRecover(position);
                    lastTapTime = -10f;
                }
                else
                {
                    player.TryGrapple(position);
                    lastTapTime = Time.time;
                }
            }
        }
    }
}
