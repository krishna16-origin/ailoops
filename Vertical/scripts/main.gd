extends Node3D

const CYAN := Color("2bd9ff")
const AMBER := Color("ffb54a")
const VOID := Color("07101a")
const INK := Color("eaf5f8")
const MAX_GRAPPLE_DISTANCE := 34.0
const TARGET_FORGIVENESS := 230.0

var player: CharacterBody3D
var camera: Camera3D
var sun: DirectionalLight3D
var environment: Environment
var rope_mesh := ImmediateMesh.new()
var rope_instance: MeshInstance3D
var targets: Array[MeshInstance3D] = []
var active_target: MeshInstance3D
var grapple_target: MeshInstance3D
var grappling := false
var gliding := false
var game_active := false
var completed := false
var checkpoint := Vector3(0, 2.2, -5.0)
var final_goal := Vector3(2.8, 66.0, 0.0)
var pointer_down := false
var pointer_origin := Vector2.ZERO
var pointer_position := Vector2.ZERO
var pointer_started := 0.0
var last_tap_ms := -1000
var steering := 0.0
var quality_index := 1
var haptics_enabled := true
var sensitivity := 1.0
var rain: GPUParticles3D
var hud: Control
var title_screen: ColorRect
var briefing_screen: ColorRect
var pause_screen: ColorRect
var complete_screen: ColorRect
var settings_screen: ColorRect
var altitude_label: Label
var state_label: Label
var toast_label: Label
var reticle: Label
var quality_label: Button
var haptics_label: Button
var sensitivity_label: Button
var toast_until := 0.0

func _ready() -> void:
	load_settings()
	build_world()
	build_player()
	build_tower()
	build_interface()
	apply_quality()
	show_title()

func _process(delta: float) -> void:
	if not game_active:
		return
	update_touch_hold()
	update_camera(delta)
	update_target_feedback()
	update_hud()
	update_rope()
	if Time.get_ticks_msec() / 1000.0 > toast_until:
		toast_label.visible = false

func _physics_process(delta: float) -> void:
	if not game_active or completed:
		return
	if grappling and is_instance_valid(grapple_target):
		simulate_swing(delta)
	else:
		simulate_air(delta)
	player.move_and_slide()
	if player.global_position.y < checkpoint.y - 12.0:
		respawn("RETURNING TO CHECKPOINT")
	if player.global_position.distance_to(final_goal) < 3.5:
		finish_chapter()
	if player.global_position.y > 26.0 and checkpoint.y < 25.0:
		checkpoint = Vector3(-2.8, 27.3, 0.0)
		show_toast("CHECKPOINT SECURED")
		pulse()
	if player.global_position.y > 47.0 and checkpoint.y < 45.0:
		checkpoint = Vector3(2.6, 48.3, 0.0)
		show_toast("UPPER SPINE SECURED")
		pulse()

func build_world() -> void:
	environment = Environment.new()
	environment.background_mode = Environment.BG_COLOR
	environment.background_color = Color("07101a")
	environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.ambient_light_color = Color("42536a")
	environment.ambient_light_energy = 0.62
	environment.tonemap_mode = Environment.TONE_MAPPER_FILMIC
	environment.glow_enabled = true
	environment.glow_intensity = 0.72
	var world_environment := WorldEnvironment.new()
	world_environment.environment = environment
	add_child(world_environment)

	sun = DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-54.0, -28.0, 0.0)
	sun.light_color = Color("9edaff")
	sun.light_energy = 1.1
	sun.shadow_enabled = true
	add_child(sun)

	var fill := OmniLight3D.new()
	fill.position = Vector3(-6.0, 25.0, -7.0)
	fill.light_color = CYAN
	fill.light_energy = 1.8
	fill.omni_range = 19.0
	add_child(fill)

	add_solid_box(Vector3(0, -1.1, 8.0), Vector3(60.0, 1.0, 120.0), facade_material(Color("0b1520")))
	build_rain()

func build_player() -> void:
	player = CharacterBody3D.new()
	player.name = "Runner"
	player.position = checkpoint
	player.floor_stop_on_slope = true
	add_child(player)
	var collision := CollisionShape3D.new()
	var shape := CapsuleShape3D.new()
	shape.radius = 0.44
	shape.height = 1.85
	collision.shape = shape
	player.add_child(collision)
	var body := MeshInstance3D.new()
	var mesh := CapsuleMesh.new()
	mesh.radius = 0.44
	mesh.height = 1.85
	body.mesh = mesh
	body.material_override = facade_material(Color("1e5066"), CYAN, 1.6)
	player.add_child(body)

	camera = Camera3D.new()
	camera.current = true
	camera.fov = 64.0
	camera.position = player.position + Vector3(0, 5.2, -14.0)
	add_child(camera)

	rope_instance = MeshInstance3D.new()
	rope_instance.mesh = rope_mesh
	var rope_material := StandardMaterial3D.new()
	rope_material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	rope_material.albedo_color = CYAN
	rope_material.emission_enabled = true
	rope_material.emission = CYAN
	rope_material.emission_energy_multiplier = 3.0
	rope_instance.material_override = rope_material
	rope_instance.visible = false
	add_child(rope_instance)

func build_tower() -> void:
	var facade := facade_material(Color("101d28"), Color("0f3042"), 0.15)
	for floor_index in range(0, 18):
		var y := float(floor_index) * 4.0
		add_solid_box(Vector3(0, y, 5.2), Vector3(11.0, 0.48, 1.1), facade)
		if floor_index % 2 == 0:
			var window_strip := make_box(Vector3(0, y + 1.1, 4.58), Vector3(9.5, 1.35, 0.12), facade_material(Color("0b1f2c"), Color("13658a"), 1.4))
			add_child(window_strip)
	for x in [-5.0, 5.0]:
		var spine := make_box(Vector3(x, 35.0, 5.0), Vector3(0.55, 72.0, 1.8), facade)
		add_child(spine)

	var ledges := [
		Vector3(0.0, 0.0, 2.8), Vector3(-2.8, 14.0, 1.1), Vector3(2.8, 26.0, 0.0),
		Vector3(-2.8, 37.0, 0.2), Vector3(2.8, 48.0, 0.0), Vector3(0.0, 59.0, 1.3), final_goal
	]
	for index in ledges.size():
		var pos: Vector3 = ledges[index]
		add_solid_box(pos, Vector3(4.3, 0.65, 3.0), facade_material(Color("193243")))
		if index == ledges.size() - 1:
			var beacon := make_box(pos + Vector3(0, 1.4, 0), Vector3(1.3, 2.3, 1.3), facade_material(Color("513615"), AMBER, 4.0))
			add_child(beacon)

	var target_positions := [
		Vector3(-2.8, 7.0, 1.6), Vector3(3.4, 13.0, 0.7), Vector3(-3.2, 20.0, 0.8),
		Vector3(3.2, 29.0, 0.8), Vector3(-3.4, 35.5, 0.9), Vector3(3.1, 42.0, 0.6),
		Vector3(-2.9, 50.0, 0.7), Vector3(2.2, 57.0, 0.5), Vector3(0.0, 64.0, 0.4)
	]
	for position in target_positions:
		create_target(position)

func build_rain() -> void:
	rain = GPUParticles3D.new()
	rain.position = Vector3(0, 38, -2)
	rain.amount = 210
	rain.lifetime = 2.4
	rain.visibility_aabb = AABB(Vector3(-16, -36, -16), Vector3(32, 72, 32))
	var material := ParticleProcessMaterial.new()
	material.direction = Vector3(0, -1, 0)
	material.spread = 12.0
	material.initial_velocity_min = 18.0
	material.initial_velocity_max = 27.0
	material.gravity = Vector3(0, -14, 0)
	material.emission_shape = ParticleProcessMaterial.EMISSION_SHAPE_BOX
	material.emission_box_extents = Vector3(14, 30, 12)
	rain.process_material = material
	var quad := QuadMesh.new()
	quad.size = Vector2(0.032, 0.72)
	var rain_material := StandardMaterial3D.new()
	rain_material.albedo_color = Color(0.3, 0.76, 1.0, 0.42)
	rain_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	rain_material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	quad.material = rain_material
	rain.draw_pass_1 = quad
	add_child(rain)

func create_target(position: Vector3) -> void:
	var target := MeshInstance3D.new()
	var mesh := SphereMesh.new()
	mesh.radius = 0.48
	mesh.height = 0.96
	target.mesh = mesh
	target.position = position
	target.material_override = facade_material(CYAN, CYAN, 3.2)
	add_child(target)
	targets.append(target)

func make_box(position: Vector3, size: Vector3, material: StandardMaterial3D) -> MeshInstance3D:
	var instance := MeshInstance3D.new()
	var mesh := BoxMesh.new()
	mesh.size = size
	instance.mesh = mesh
	instance.position = position
	instance.material_override = material
	return instance

func add_solid_box(position: Vector3, size: Vector3, material: StandardMaterial3D) -> void:
	var visual := make_box(position, size, material)
	add_child(visual)
	var solid := StaticBody3D.new()
	solid.position = position
	var collider := CollisionShape3D.new()
	var shape := BoxShape3D.new()
	shape.size = size
	collider.shape = shape
	solid.add_child(collider)
	add_child(solid)

func facade_material(albedo: Color, emission: Color = Color.BLACK, energy := 0.0) -> StandardMaterial3D:
	var material := StandardMaterial3D.new()
	material.albedo_color = albedo
	material.metallic = 0.55
	material.roughness = 0.42
	if energy > 0.0:
		material.emission_enabled = true
		material.emission = emission
		material.emission_energy_multiplier = energy
	return material

func simulate_air(delta: float) -> void:
	player.velocity.y -= 22.0 * delta
	if gliding and player.velocity.y < 2.5:
		player.velocity.y = max(player.velocity.y, -3.1)
		player.velocity += Vector3(steering * 4.0, 2.0, 5.2) * delta
	else:
		player.velocity += Vector3(steering * 2.2, 0, 0) * delta
	player.velocity.x = clamp(player.velocity.x, -15.0, 15.0)
	player.velocity.z = clamp(player.velocity.z, -3.0, 17.0)

func simulate_swing(delta: float) -> void:
	if not is_instance_valid(grapple_target):
		release_grapple(Vector2.ZERO)
		return
	var anchor := grapple_target.global_position
	var offset := player.global_position - anchor
	var length := maxf(offset.length(), 0.1)
	var rope_length := minf(MAX_GRAPPLE_DISTANCE, 10.5)
	var radial := offset / length
	var radial_velocity := player.velocity.dot(radial)
	if radial_velocity > 0.0:
		player.velocity -= radial * radial_velocity
	player.velocity += Vector3(steering * 15.0, -20.0, 4.5) * delta
	if length > rope_length:
		player.velocity -= radial * ((length - rope_length) / maxf(delta, 0.01))
	player.velocity = player.velocity.limit_length(33.0)

func update_camera(delta: float) -> void:
	var desired := player.global_position + Vector3(0, 5.4 if not gliding else 6.7, -14.0 if not gliding else -17.0)
	camera.global_position = camera.global_position.lerp(desired, minf(delta * 5.2, 1.0))
	camera.look_at(player.global_position + Vector3(0, 1.0, 4.0), Vector3.UP)
	camera.fov = lerpf(camera.fov, 70.0 if gliding else 64.0, minf(delta * 4.0, 1.0))

func update_target_feedback() -> void:
	if grappling:
		reticle.visible = false
		return
	active_target = find_best_target(pointer_position if pointer_down else get_viewport().get_visible_rect().size * 0.5)
	reticle.visible = active_target != null
	if active_target != null:
		var point := camera.unproject_position(active_target.global_position)
		reticle.position = point - Vector2(20, 20)
	for target in targets:
		var material := target.material_override as StandardMaterial3D
		material.emission_energy_multiplier = 8.0 if target == active_target else 3.2

func find_best_target(screen_point: Vector2) -> MeshInstance3D:
	var best: MeshInstance3D
	var best_score := INF
	for target in targets:
		if not is_instance_valid(target):
			continue
		var offset := target.global_position - player.global_position
		var distance := offset.length()
		if distance > MAX_GRAPPLE_DISTANCE or target.global_position.y < player.global_position.y - 1.5:
			continue
		var point := camera.unproject_position(target.global_position)
		if point.x < -100.0 or point.x > get_viewport().get_visible_rect().size.x + 100.0:
			continue
		var screen_distance := point.distance_to(screen_point)
		if screen_distance > TARGET_FORGIVENESS:
			continue
		var score := screen_distance + distance * 1.2 - maxf(0.0, offset.y) * 3.0
		if score < best_score:
			best = target
			best_score = score
	return best

func try_grapple(screen_point: Vector2) -> void:
	if not game_active or completed:
		return
	var target := find_best_target(screen_point)
	if target == null:
		show_toast("NO GRAPPLE LOCK")
		return
	grapple_target = target
	grappling = true
	gliding = false
	rope_instance.visible = true
	state_label.text = "RELEASE TO LAUNCH"
	show_toast("GRAPPLE LOCKED")
	pulse()

func release_grapple(drag: Vector2) -> void:
	if not grappling:
		return
	grappling = false
	gliding = false
	rope_instance.visible = false
	grapple_target = null
	player.velocity += Vector3(drag.x * 0.026, maxf(2.6, drag.y * -0.012), 3.4)
	state_label.text = "HOLD TO GLIDE"
	pulse()

func update_rope() -> void:
	if not grappling or not is_instance_valid(grapple_target):
		return
	rope_mesh.clear_surfaces()
	rope_mesh.surface_begin(Mesh.PRIMITIVE_LINES)
	rope_mesh.surface_add_vertex(player.global_position + Vector3.UP * 0.55)
	rope_mesh.surface_add_vertex(grapple_target.global_position)
	rope_mesh.surface_end()

func update_touch_hold() -> void:
	if not pointer_down or grappling:
		return
	steering = clampf((pointer_position.x - pointer_origin.x) / (get_viewport().get_visible_rect().size.x * 0.22) * sensitivity, -1.0, 1.0)
	if Time.get_ticks_msec() / 1000.0 - pointer_started > 0.19:
		gliding = true
		state_label.text = "GLIDE // HOLD"

func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("pause") and game_active:
		toggle_pause()
	if not game_active or pause_screen.visible or settings_screen.visible:
		return
	if event is InputEventScreenTouch:
		handle_touch(event.position, event.pressed)
	elif event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT:
		handle_touch(event.position, event.pressed)

func handle_touch(position: Vector2, pressed: bool) -> void:
	if pressed:
		pointer_down = true
		pointer_origin = position
		pointer_position = position
		pointer_started = Time.get_ticks_msec() / 1000.0
		return
	pointer_position = position
	var elapsed := Time.get_ticks_msec() / 1000.0 - pointer_started
	pointer_down = false
	steering = 0.0
	if grappling:
		release_grapple(position - pointer_origin)
		return
	if gliding:
		gliding = false
		state_label.text = "TAP CYAN TARGET"
		return
	if elapsed < 0.25:
		var now := Time.get_ticks_msec()
		if now - last_tap_ms < 340:
			try_grapple(position)
		last_tap_ms = now
		try_grapple(position)

func build_interface() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)
	hud = Control.new()
	hud.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	hud.mouse_filter = Control.MOUSE_FILTER_IGNORE
	layer.add_child(hud)
	altitude_label = create_label("ALT 000m", Vector2(48, 46), Vector2(310, 58), 28, INK)
	state_label = create_label("TAP CYAN TARGET", Vector2(235, 1715), Vector2(610, 60), 27, CYAN)
	state_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	var pause_button := create_button("Ⅱ", Vector2(930, 42), Vector2(82, 66), toggle_pause, false)
	pause_button.add_theme_font_size_override("font_size", 34)
	reticle = create_label("◉", Vector2.ZERO, Vector2(48, 48), 45, CYAN)
	reticle.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	reticle.visible = false
	toast_label = create_label("", Vector2(90, 390), Vector2(900, 56), 25, AMBER)
	toast_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	toast_label.visible = false

	title_screen = create_screen()
	create_label("// STORM PROTOCOL", Vector2(190, 430), Vector2(700, 50), 28, CYAN, title_screen).horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	var title := create_label("VERTICAL", Vector2(70, 555), Vector2(940, 150), 104, INK, title_screen)
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	var premise := create_label("THE TOWER REMEMBERS WHAT THEY BURIED.", Vector2(100, 735), Vector2(880, 80), 23, Color("a7bac2"), title_screen)
	premise.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	create_button("START ASCENT", Vector2(185, 1050), Vector2(710, 96), show_briefing, true, title_screen)
	quality_label = create_button("", Vector2(185, 1172), Vector2(710, 82), cycle_quality, false, title_screen)
	create_button("SETTINGS", Vector2(185, 1278), Vector2(710, 82), show_settings, false, title_screen)
	var instruction := create_label("Tap cyan targets. Release to launch. Hold in air to glide.", Vector2(90, 1540), Vector2(900, 74), 22, Color("93aab5"), title_screen)
	instruction.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER

	briefing_screen = create_screen()
	create_label("CHAPTER 01 // THE SERVICE SPINE", Vector2(90, 405), Vector2(900, 72), 29, CYAN, briefing_screen).horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	var brief := create_label("Two years after the collapse, the tower still carries the scars. Find the amber maintenance ledge and recover the first authorization trace.", Vector2(125, 640), Vector2(830, 270), 32, INK, briefing_screen)
	brief.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	brief.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	create_button("BEGIN", Vector2(185, 1150), Vector2(710, 96), begin_mission, true, briefing_screen)

	pause_screen = create_screen()
	create_label("ASCENT PAUSED", Vector2(120, 590), Vector2(840, 100), 52, INK, pause_screen).horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	create_button("RESUME", Vector2(185, 910), Vector2(710, 90), toggle_pause, true, pause_screen)
	create_button("RESTART CHECKPOINT", Vector2(185, 1025), Vector2(710, 90), restart_mission, false, pause_screen)
	create_button("RETURN TO TITLE", Vector2(185, 1140), Vector2(710, 90), show_title, false, pause_screen)

	complete_screen = create_screen()
	create_label("ACCESS TRACE RECOVERED", Vector2(70, 430), Vector2(940, 100), 45, AMBER, complete_screen).horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	var ending := create_label("The signature is real. Someone approved the substitution, then sealed the record behind the executive floors.", Vector2(130, 660), Vector2(820, 260), 31, INK, complete_screen)
	ending.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	ending.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	create_button("REPLAY CHAPTER", Vector2(185, 1120), Vector2(710, 90), restart_mission, true, complete_screen)
	create_button("RETURN TO TITLE", Vector2(185, 1235), Vector2(710, 90), show_title, false, complete_screen)

	settings_screen = create_screen()
	create_label("FIELD SETTINGS", Vector2(120, 405), Vector2(840, 100), 50, INK, settings_screen).horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	haptics_label = create_button("", Vector2(185, 700), Vector2(710, 85), toggle_haptics, false, settings_screen)
	sensitivity_label = create_button("", Vector2(185, 810), Vector2(710, 85), cycle_sensitivity, false, settings_screen)
	create_button("BACK", Vector2(185, 1090), Vector2(710, 90), close_settings, true, settings_screen)
	update_setting_labels()

func create_screen() -> ColorRect:
	var screen := ColorRect.new()
	screen.color = Color(0.025, 0.04, 0.06, 0.96)
	screen.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	screen.visible = false
	hud.add_child(screen)
	return screen

func create_label(text_value: String, position_value: Vector2, size_value: Vector2, font_size: int, color: Color, parent: Control = null) -> Label:
	var label := Label.new()
	label.text = text_value
	label.position = position_value
	label.size = size_value
	label.add_theme_font_size_override("font_size", font_size)
	label.add_theme_color_override("font_color", color)
	label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	(parent if parent != null else hud).add_child(label)
	return label

func create_button(text_value: String, position_value: Vector2, size_value: Vector2, callback: Callable, primary: bool, parent: Control = null) -> Button:
	var button := Button.new()
	button.position = position_value
	button.size = size_value
	button.text = text_value
	button.add_theme_font_size_override("font_size", 29)
	button.add_theme_color_override("font_color", VOID if primary else INK)
	button.add_theme_color_override("font_hover_color", VOID if primary else CYAN)
	button.add_theme_color_override("font_pressed_color", VOID if primary else CYAN)
	button.pressed.connect(callback)
	(parent if parent != null else hud).add_child(button)
	return button

func show_title() -> void:
	game_active = false
	get_tree().paused = false
	reset_screens()
	title_screen.visible = true

func show_briefing() -> void:
	reset_screens()
	briefing_screen.visible = true

func begin_mission() -> void:
	reset_screens()
	game_active = true
	completed = false
	respawn("SERVICE SPINE // REACH THE AMBER LEDGE")

func restart_mission() -> void:
	get_tree().paused = false
	reset_screens()
	game_active = true
	completed = false
	respawn("ASCENT RESTARTED")

func toggle_pause() -> void:
	if not game_active:
		return
	var next_pause := not get_tree().paused
	get_tree().paused = next_pause
	pause_screen.visible = next_pause

func show_settings() -> void:
	title_screen.visible = false
	settings_screen.visible = true
	update_setting_labels()

func close_settings() -> void:
	settings_screen.visible = false
	title_screen.visible = true

func reset_screens() -> void:
	title_screen.visible = false
	briefing_screen.visible = false
	pause_screen.visible = false
	complete_screen.visible = false
	settings_screen.visible = false

func respawn(message: String) -> void:
	grappling = false
	gliding = false
	rope_instance.visible = false
	grapple_target = null
	player.global_position = checkpoint
	player.velocity = Vector3.ZERO
	state_label.text = "TAP CYAN TARGET"
	show_toast(message)

func finish_chapter() -> void:
	completed = true
	game_active = false
	grappling = false
	rope_instance.visible = false
	get_tree().paused = false
	complete_screen.visible = true
	show_toast("TRACE RECOVERED")
	pulse()

func update_hud() -> void:
	altitude_label.text = "ALT %03dm" % maxf(0.0, player.global_position.y)
	if not grappling and not gliding:
		state_label.text = "TAP CYAN TARGET"

func show_toast(message: String) -> void:
	toast_label.text = message
	toast_label.visible = true
	toast_until = Time.get_ticks_msec() / 1000.0 + 2.2

func pulse() -> void:
	if haptics_enabled:
		Input.vibrate_handheld(26, 0.5)

func cycle_quality() -> void:
	quality_index = (quality_index + 1) % 3
	apply_quality()
	save_settings()

func apply_quality() -> void:
	var profiles := ["PERFORMANCE", "BALANCED", "CINEMATIC"]
	quality_label.text = "GRAPHICS // " + profiles[quality_index]
	if rain != null:
		rain.amount = [95, 210, 420][quality_index]
	if sun != null:
		sun.shadow_enabled = quality_index == 2
	if environment != null:
		environment.glow_enabled = quality_index > 0
		environment.glow_intensity = [0.25, 0.72, 1.05][quality_index]

func toggle_haptics() -> void:
	haptics_enabled = not haptics_enabled
	save_settings()
	update_setting_labels()

func cycle_sensitivity() -> void:
	sensitivity = 0.8 if sensitivity >= 1.3 else (1.0 if sensitivity < 1.0 else 1.3)
	save_settings()
	update_setting_labels()

func update_setting_labels() -> void:
	if haptics_label != null:
		haptics_label.text = "HAPTICS // " + ("ON" if haptics_enabled else "OFF")
	if sensitivity_label != null:
		sensitivity_label.text = "SWIPE SENSITIVITY // %.1fx" % sensitivity

func save_settings() -> void:
	var config := ConfigFile.new()
	config.set_value("settings", "quality", quality_index)
	config.set_value("settings", "haptics", haptics_enabled)
	config.set_value("settings", "sensitivity", sensitivity)
	config.save("user://vertical_settings.cfg")

func load_settings() -> void:
	var config := ConfigFile.new()
	if config.load("user://vertical_settings.cfg") == OK:
		quality_index = int(config.get_value("settings", "quality", 1))
		haptics_enabled = bool(config.get_value("settings", "haptics", true))
		sensitivity = float(config.get_value("settings", "sensitivity", 1.0))
