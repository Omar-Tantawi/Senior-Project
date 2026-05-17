<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Http\Requests\VacationRequest\UpdateVacationStatusRequest;
use App\Models\Admin;
use App\Models\VacationRequest;
use App\Repositories\VacationRequestRepository;
use Illuminate\Http\Request;

class VacationRequestController extends Controller
{
    public function __construct(private VacationRequestRepository $repo) {}

    public function index(Request $request)
    {
        return response()->json(
            $this->repo->filter(
                $request->only(['teacher_id', 'status', 'from', 'to']),
                $request->input('per_page', 20)
            )
        );
    }

    public function show(int $id)
    {
        return response()->json(
            VacationRequest::where('vacation_id', $id)
                ->with(['teacher.user', 'approvedByAdmin.user'])
                ->firstOrFail()
        );
    }

    public function update(UpdateVacationStatusRequest $request, int $id)
    {
        $vacation = VacationRequest::where('vacation_id', $id)->firstOrFail();
        $admin    = Admin::where('user_id', auth()->id())->firstOrFail();

        $vacation->update([
            'status'             => $request->validated()['status'],
            'approvedbyadmin_id' => $admin->admin_id,
        ]);

        return response()->json($vacation->load(['teacher.user', 'approvedByAdmin.user']));
    }

    public function destroy(int $id)
    {
        VacationRequest::where('vacation_id', $id)->firstOrFail()->delete();
        return response()->json(['message' => 'Vacation request deleted successfully.']);
    }
}
