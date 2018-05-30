var server_url = 'http://127.0.0.1:8000/miptapp/';

var first_image_box = $("#first_image_box");
var second_image_box = $("#second_image_box");

$(document).ready( function () {
    first_image_box.hide();
    second_image_box.hide();
});

$("#submit_button").click( function(){
    var fd = new FormData();
    fd.append('img', $("#input_form").prop('files')[0]);
    $.ajax({
        url: server_url + 'upload/',
        data: fd,
        cache: false,
        contentType: false,
        processData: false,
        method: 'POST',
        success: function(data){
            $("#first_image").attr('src', 'mipt/'+data.input_image);
            $("#second_image").attr('src', 'mipt/'+data.output_image);
            first_image_box.show();
            second_image_box.show();
        }
    });
});




