﻿(function($) {
    $.fn.extend({
        collapsiblePanel: function() {
            // Call the ConfigureCollapsiblePanel function for the selected element
            return $(this).each(ConfigureCollapsiblePanel);
        }
    });

})(jQuery);

function ConfigureCollapsiblePanel() {
    $(this).addClass("ui-widget");

    // Check if there are any child elements, if not then wrap the inner text within a new div.
    if ($(this).children().length == 0) {
        $(this).wrapInner("<div></div>");
    }

    // Wrap the contents of the container within a new div.
    $(this).children().wrapAll("<div class='collapsibleContainerContent ui-widget-content'></div>");

    // Create a new div as the first item within the container.  Put the title of the panel in here.
    $("<div class='collapsibleContainerTitle ui-widget-header'><div>" + $(this).attr("title") + "</div></div>").prependTo($(this));

    // Assign a call to CollapsibleContainerTitleOnClick for the click event of the new title div.
    $(".collapsibleContainerTitle", this).click(CollapsibleContainerTitleOnClick);

	//start off in collapsed state
	// CollapsibleContainerTitleOnClick();
}

function CollapsibleContainerTitleOnClick() {
    // The item clicked is the title div... get this parent (the overall container) and toggle the content within it.
    $(".collapsibleContainerContent", $(this).parent()).slideToggle();
}