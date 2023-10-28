function fncSoundPlay(fileName)
{
	var audio = new Audio("C:/Users/lixxc/PycharmProjects/cbnu_kioskAi/"+fileName);
	audio.load();
	audio.volume = 1;
	audio.play();
	audio.autoplay = true;
}