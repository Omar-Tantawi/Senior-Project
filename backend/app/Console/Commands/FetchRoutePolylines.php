<?php

namespace App\Console\Commands;

use App\Models\BusRoute;
use Illuminate\Console\Command;
use Illuminate\Support\Facades\Http;

/**
 * One-shot: fetches road-snapped polylines for every route from the public
 * OSRM demo server and persists them on the route row.
 *
 *   php artisan transport:fetch-polylines
 *   php artisan transport:fetch-polylines --route=1
 *   php artisan transport:fetch-polylines --force   # re-fetch even if cached
 *
 * After this runs once, the DB has everything it needs — the demo provider
 * works offline forever.
 */
class FetchRoutePolylines extends Command
{
    protected $signature = 'transport:fetch-polylines
                            {--route= : Only this route_id}
                            {--force  : Overwrite existing polylines}';

    protected $description = 'Fetch road-snapped polylines from OSRM and store them per route.';

    private const OSRM_BASE = 'https://router.project-osrm.org/route/v1/driving/';

    public function handle(): int
    {
        $query = BusRoute::with(['stops' => fn ($q) => $q->orderBy('stoporder')]);
        if ($id = $this->option('route')) {
            $query->where('route_id', (int) $id);
        }
        $routes = $query->get();

        if ($routes->isEmpty()) {
            $this->warn('No routes found.');
            return self::SUCCESS;
        }

        $force = (bool) $this->option('force');

        foreach ($routes as $route) {
            $this->info("Route #{$route->route_id} — {$route->name}");

            if ($route->polyline && ! $force) {
                $this->line('   already has a polyline (skip — pass --force to refresh)');
                continue;
            }

            $stops = $route->stops->filter(
                fn ($s) => $s->latitude !== null && $s->longitude !== null,
            )->values();

            if ($stops->count() < 2) {
                $this->warn('   needs at least 2 geocoded stops, skipping');
                continue;
            }

            // OSRM expects coordinates as `lng,lat;lng,lat;…`
            $coords = $stops->map(fn ($s) => $s->longitude . ',' . $s->latitude)
                ->implode(';');

            $url = self::OSRM_BASE . $coords
                . '?overview=full&geometries=geojson';

            try {
                $resp = Http::timeout(20)->get($url);
            } catch (\Throwable $e) {
                $this->error('   network error: ' . $e->getMessage());
                continue;
            }

            if (! $resp->ok()) {
                $this->error('   OSRM returned HTTP ' . $resp->status());
                continue;
            }

            $json = $resp->json();
            $coordsOut = $json['routes'][0]['geometry']['coordinates'] ?? null;

            if (! is_array($coordsOut) || count($coordsOut) < 2) {
                $this->error('   no usable geometry in response');
                continue;
            }

            // OSRM returns [lng, lat]; we store as [lat, lng] to match the rest
            // of the codebase.
            $polyline = array_map(
                fn ($p) => [round((float) $p[1], 7), round((float) $p[0], 7)],
                $coordsOut,
            );

            $route->polyline = $polyline;
            $route->save();

            $this->info('   stored ' . count($polyline) . ' waypoints');
        }

        $this->newLine();
        $this->info('Done.');
        return self::SUCCESS;
    }
}
