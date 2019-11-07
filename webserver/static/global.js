var g_labels;
const ID_BUTTON_AS_PREDICTED = -1;
const BUTTON_NEGATIVE = 0;
const BUTTON_POSITIVE = 1;
const COLOR_CERTAIN = '#40bf40';
const COLOR_UNCERTAIN = '#d2d2d2';

const NORMAL_TRAINING = 1;
const SOFT_TRAINING = 2;
const HARD_TRAINING = 4;


function get_bg_color(idx) {

	var bg_color;
	switch(idx) {
		case BUTTON_NEGATIVE:
			bg_color = '#FFB890';
			break;
		case BUTTON_POSITIVE:
			bg_color = '#8DB1E6';
			break;
		case ID_BUTTON_AS_PREDICTED:
			bg_color = '#ffffff';
			break;
		default:
			bg_color = '#ffffff';
			break;
	}

	return bg_color;
}

function get_bd_color(idx) {
	var bd_color;
	switch(idx) {
		case BUTTON_NEGATIVE:
			bd_color = '#D31600';
			break;
		case BUTTON_POSITIVE:
			bd_color = '#1E4B8F';
			break;
		case ID_BUTTON_AS_PREDICTED:
			bd_color = '#000000';
			break;
		default:
			bd_color = '#000000';
			break;
	}

	return bd_color;
}