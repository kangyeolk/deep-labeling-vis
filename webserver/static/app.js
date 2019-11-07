function unpack(rows, key) {
	return rows.map(function(row) { return row[key]; });
}

function initUI() {
	$('.loading').on('click', function() {
		var $this = $(this);
		$this.button('loading');
		// setTimeout(function() {
		//     $this.button('reset');
		// }, 3000);
	});

	$("#submit_btn").click(function() {

	});

	$("#init_project").click(function() {
		console.log("init_project clicked")
		// window.socket.emit("request_init_project", { data: 'test'});
		window.socket.emit("request_init_project", { 'panel_width': PANEL_WIDTH, 'patch_size': PATCH_SIZE});
	});
}

function on_init_project(param) {
	var obj_json = JSON.parse(param);
	sentence_vector = obj_json['sentence_vector'];
	// set_sentence_vector(sentence_vector);

	// create embedding view
	// tsne_init(tsne_getData());
	// tsne_run();
}

function on_recv_pass_label_info_received(param) {
	// var obj_json = JSON.parse(param);

	alert(param);
}

function on_recv_patches_info(param) {
	// console.log(param);

	var obj_json = JSON.parse(param);
	var image_size_width = obj_json['image_size'][0];
	var patch_info = obj_json['patches_info'];

	console.log(image_size_width);
	lv0 = patch_info['0'];
	console.log(lv0);

	var patch_width = PANEL_WIDTH * PATCH_SIZE / image_size_width;
	var x, y;
	for (y in lv0) {
		for (x in lv0[y]) {
			var width = patch_width;
			var height = patch_width;
			var pos_x = x * patch_width;
			var pos_y = y * patch_width;
			
			var rect = document.createElement('div');
			rect.setAttribute("id", "patch_"+y+"_"+x);
			rect.setAttribute("class", "patch_class");
			
			console.log(lv0[y][x]['label'])
			if (lv0[y][x]['label'] == '' || lv0[y][x]['label'] == 'bg') {
			// if (lv0[y][x]['label'] == '') {

			} else {
				var color = "black";
				if (lv0[y][x]['label'] == 'normal') {
					color = "#000000";
				} else if (lv0[y][x]['label'] == 'hp') {
					color = "#66FF66";
				} else if (lv0[y][x]['label'] == 'ta') {
					color = "#FF0000";	
				}
				rect.setAttribute("style","opacity: 1; position: absolute; cursor: default; width: "+width+"px; height: "+height+"px; left: "+pos_x+"px; top: "+pos_y+"px; z-index: 100; border: 1px; border-style: solid; border-color: "+color+";\"");			
			}

			var c = document.getElementById("whole_image_panel");
			c.insertBefore(rect, c.childNodes[0]);
		}
	}
	
}

$(document).ready(function() {
	// Init websocket
	window.socket = io.connect('http://' + document.domain + ':' + location.port + '/title_temp');
	var socket = window.socket;
	// console.log(socket);
	var tsne_div = document.getElementById("scatter_div_a");
	var tsne_div_b = document.getElementById("scatter_div_b");

	socket.on('server_response', function(msg) {
		// $('#log').append('<p>Received: ' + msg.data + '</p>');
		console.log(msg);

		if (msg.id === 'connect') {
			if (msg.data.info === 'new') {}

		} else if (msg.id === 'message') {

			if (msg.data.startsWith('Error')) {
				$.notify(msg.data, "error");
				$('.loading').button('reset');

			} else if(msg.data.startsWith('init_project_complete')) {
				console.log('init_project_complete')
				on_init_project(msg.param);

			} else if(msg.data.startsWith('pass_label_info_received')) {
			    on_recv_pass_label_info_received(msg.param);
			} else if(msg.data.startsWith('patches_info')) {
				on_recv_patches_info(msg.param);
			}else {
				$.notify(msg.data, "success");
			}
		}
	});

	initUI();
});