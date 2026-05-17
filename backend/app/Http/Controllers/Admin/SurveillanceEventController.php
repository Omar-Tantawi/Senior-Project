<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Http\Requests\SurveillanceEvent\SurveillanceSummaryRequest;
use App\Http\Requests\SurveillanceEvent\UpdateSurveillanceStatusRequest;
use App\Models\SurveillanceEvent;
use App\Repositories\SurveillanceEventRepository;
use Illuminate\Http\Request;

class SurveillanceEventController extends Controller
{
    public function __construct(private SurveillanceEventRepository $repo) {}

    public function index(Request $request)
    {
        return response()->json(
            $this->repo->filter(
                $request->only(['camera_id', 'detectedtype', 'severity', 'student_id', 'section_id', 'status', 'from', 'to']),
                $request->input('per_page', 20)
            )
        );
    }

    public function show(int $id)
    {
        return response()->json(
            SurveillanceEvent::where('survevent_id', $id)
                ->with(['camera', 'student.user', 'section.schoolClass', 'assessment'])
                ->firstOrFail()
        );
    }

    public function summary(SurveillanceSummaryRequest $request)
    {
        $data   = $request->validated();
        $events = $this->repo->summaryBetween($data['from'], $data['to'], $request->filled('camera_id') ? $request->camera_id : null);

        return response()->json([
            'period'      => ['from' => $data['from'], 'to' => $data['to']],
            'total'       => $events->count(),
            'by_type'     => $events->groupBy('detectedtype')->map->count(),
            'by_severity' => $events->groupBy('severity')->map->count(),
            'by_camera'   => $events->groupBy('camera_id')->map->count(),
        ]);
    }

    public function updateStatus(UpdateSurveillanceStatusRequest $request, int $id)
    {
        $event = SurveillanceEvent::where('survevent_id', $id)->firstOrFail();
        $event->update(['status' => $request->validated()['status']]);
        return response()->json($event);
    }

    public function footage(string $filename)
    {
        $dir      = rtrim(env('KIRA_FOOTAGE_PATH', ''), '/\\');
        $filepath = $dir . DIRECTORY_SEPARATOR . $filename;

        abort_unless(file_exists($filepath), 404, 'Footage file not found.');

        return response()->download($filepath, $filename);
    }

    public function destroy(int $id)
    {
        SurveillanceEvent::where('survevent_id', $id)->firstOrFail()->delete();
        return response()->json(['message' => 'Surveillance event deleted successfully.']);
    }
}
