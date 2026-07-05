def configure_routing_only(dispatcher):
    """
    Configures the TaskDispatcher in Routing-Only mode.
    Complexity routing and agent pruning are active.
    Context caching is disabled (forces full prompt prefill on every step).
    """
    dispatcher.routing_disabled = False
    dispatcher.cache_disabled = True
    dispatcher.baseline_name = "routing_only"

def configure_cache_only(dispatcher):
    """
    Configures the TaskDispatcher in Cache-Only mode.
    Complexity routing is disabled (statically maps all tasks to Large workstation tier, all agents active).
    Context caching is fully active.
    """
    dispatcher.routing_disabled = True
    dispatcher.cache_disabled = False
    dispatcher.baseline_name = "cache_only"
