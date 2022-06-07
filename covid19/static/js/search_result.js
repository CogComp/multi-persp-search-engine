$(document).ready(function(){
    $("badge.source-selector").click(toggleHardClassify);
    $("badge.type-selector").click(toggleHardClassify);
    $("badge.topic-selector").click(toggleSoftClassify);

    // Function to run when a given selector can have only when category (i.e. source type, document type)
    function toggleHardClassify() {
        toggleBadgeClasses(this);
        let type = this.id;
        $("div." + type).toggle();
    };

    // Function to run when selector can have more than one category (i.e. topic)
    function toggleSoftClassify() {
        toggleBadgeClasses(this);
        let type = this.id;
        $("div." + type).each(function() {
            let classes = this.classList;
            for (i = 0; i < classes.length; ++i) {
                let currentClass = classes[i];
                if ($("badge.topic-selector#" + currentClass).hasClass("on")) {
                    $(this).show();
                    return true;
                }
            }
            $(this).hide();
        });
    };

    // Handles the changing of color and labeling the badge on or off.
    function toggleBadgeClasses(badge) {
        let default_tag_color_class = "palette-nephritis";
        let disable_color_class = "palette-concrete";

        if ($(badge).hasClass("source-selector")) {
            default_tag_color_class = "palette-nephritis";
        }
        else if ($(badge).hasClass("type-selector")) {
            default_tag_color_class = "palette-peter-river";
        }
        else if ($(badge).hasClass("topic-selector")) {
            default_tag_color_class = "palette-belize-hole";
        }

        if ($(badge).hasClass("on")) {
            $(badge).addClass(disable_color_class).removeClass(default_tag_color_class);
            $(badge).addClass("off").removeClass("on");
        } else {
            $(badge).addClass(default_tag_color_class).removeClass(disable_color_class);
            $(badge).addClass("on").removeClass("off");
        }
    }

    $(".toggle").click(function() {
        if ($(this).find('i').hasClass("up")) {
            $(this).find('i').addClass("down").removeClass("up");
        } else {
            $(this).find('i').addClass("up").removeClass("down")
        }
    });

    $(".blockquote").click(function() {
        let quote = $(this).find('p').text()
        window.location.href = "/quote/?q=" + encodeURIComponent(quote);
    })

    // Keep track of number of articles to display based on screen size.
    let width;
    let numArticles;
    setInterval(() => onResize(), 300);

    function onResize() {
        width = $(window).width();
        let oldNum = numArticles;
        if (width >= 748) {
            numArticles = 2;
        } else {
            numArticles = 1;
        }
        if (width < 576) {
            $(".arrow").hide();
        } else {
            $(".arrow").show();
        }
        if (oldNum !== numArticles) {
            redraw(numArticles);
        }
    }

    function redraw(numArticles) {
        $(".previous").each(function() {
            let $id = this.id.split("-");
            let $source = $id[0];
            let $position = $id[1];
            if (parseInt($position) === 0) {
                $(this).css('visibility', 'hidden');
            } else {
                $(this).css('visibility', 'visible');
            }
            if ($(".article." + $source + "." + (parseInt($position) + numArticles).toString()).length === 0) {
                $("#" + $source + "-" + $position + ".next").css('visibility', 'hidden');
            } else {
                $("#" + $source + "-" + $position + ".next").css('visibility', 'visible');
            }
            $(".article." + $source).hide();
            for (let i = 0; i < numArticles; i++) {
                $(".article." + $source + "." + (parseInt($position) + i).toString()).show();
            }
        });
    }

    $(".next").click(function() {
        let $id = this.id.split("-");
        let $source = $id[0];
        let $oldPosition = $id[1];
        let $newPosition = (parseInt($oldPosition)+numArticles).toString();
        $(this).attr('id', $source + "-" + $newPosition);
        $("#" + $source + "-" + $oldPosition).attr('id', $source + "-" + $newPosition);
        for (let i = 0; i < numArticles; i++) {
            $(".article." + $source + "." + (parseInt($newPosition) + i).toString()).show();
            $(".article." + $source + "." + (parseInt($oldPosition) + i).toString()).hide();
        }
        if ($(".article." + $source + "." + (parseInt($newPosition)+numArticles).toString()).length === 0) {
            $(this).css('visibility','hidden');
        }
        $("#" + $source + "-" + $newPosition + ".previous").css('visibility','visible');
    });

    $(".previous").click(function() {
        let $id = this.id.split("-");
        let $source = $id[0];
        let $oldPosition = $id[1];
        let $newPosition = (Math.max(0, parseInt($oldPosition)-numArticles)).toString();
        $(this).attr('id', $source + "-" + $newPosition);
        $("#" + $source + "-" + $oldPosition).attr('id', $source + "-" + $newPosition);
        for (let i = 0; i < numArticles; i++) {
            $(".article." + $source + "." + (parseInt($newPosition) + i).toString()).show();
            $(".article." + $source + "." + (parseInt($oldPosition) + i).toString()).hide();
        }
        if (parseInt($newPosition) === 0) {
            $(this).css('visibility','hidden');
        }
        $("#" + $source + "-" + $newPosition + ".next").css('visibility','visible');
    });
});