<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Http\Requests\StudentFeePlan\StoreStudentFeePlanRequest;
use App\Http\Requests\StudentFeePlan\UpdateStudentFeePlanRequest;
use App\Models\StudentFeePlan;
use App\Services\StudentFeePlanService;
use Illuminate\Http\Request;

class StudentFeePlanController extends Controller
{
    public function __construct(private StudentFeePlanService $service) {}

    public function byStudent(Request $request)
    {
        $query = StudentFeePlan::with(['student.user', 'feePlan.schoolYear']);

        if ($request->filled('schoolyear_id')) {
            $query->whereHas('feePlan', fn($q) => $q->where('schoolyear_id', $request->schoolyear_id));
        }

        $all = $query->get();

        $grouped = $all->groupBy('student_id')->map(function ($accounts) {
            $student   = $accounts->first()->student;
            $totalFee  = $accounts->sum(fn($a) => (float) optional($a->feePlan)->totalamount);
            $totalPaid = $accounts->sum(fn($a) => (float) $a->paid_amount);
            $totalBal  = $accounts->sum(fn($a) => (float) $a->balance);

            $overallStatus = 'unpaid';
            if ($totalPaid >= $totalFee && $totalFee > 0) $overallStatus = 'paid';
            elseif ($totalPaid > 0)                        $overallStatus = 'partial';

            return [
                'student_id'     => $accounts->first()->student_id,
                'student_name'   => $student?->user?->name ?? "#{$accounts->first()->student_id}",
                'plans'          => $accounts->map(fn($a) => [
                    'account_id'  => $a->account_id,
                    'feeplan_id'  => $a->feeplan_id,
                    'plan_name'   => optional($a->feePlan)->name,
                    'school_year' => optional(optional($a->feePlan)->schoolYear)->name,
                    'total'       => (float) optional($a->feePlan)->totalamount,
                    'paid'        => (float) $a->paid_amount,
                    'balance'     => (float) $a->balance,
                    'status'      => $a->status,
                    'due_date'    => $a->due_date?->format('Y-m-d'),
                    'notes'       => $a->notes,
                ])->values(),
                'total_fee'      => $totalFee,
                'total_paid'     => $totalPaid,
                'total_balance'  => $totalBal,
                'overall_status' => $overallStatus,
            ];
        })->values();

        if ($request->filled('status')) {
            $grouped = $grouped->filter(fn($s) => $s['overall_status'] === $request->status)->values();
        }

        if ($request->filled('search')) {
            $term    = strtolower($request->search);
            $grouped = $grouped->filter(fn($s) => str_contains(strtolower($s['student_name']), $term))->values();
        }

        $perPage = (int) $request->input('per_page', 20);
        $page    = (int) $request->input('page', 1);
        $total   = $grouped->count();
        $items   = $grouped->slice(($page - 1) * $perPage, $perPage)->values();

        return response()->json([
            'data'         => $items,
            'total'        => $total,
            'per_page'     => $perPage,
            'current_page' => $page,
        ]);
    }

    public function index(Request $request)
    {
        $query = StudentFeePlan::with(['student.user', 'feePlan.schoolYear']);

        if ($request->filled('student_id'))   $query->where('student_id', $request->student_id);
        if ($request->filled('feeplan_id'))    $query->where('feeplan_id', $request->feeplan_id);
        if ($request->filled('schoolyear_id')) $query->whereHas('feePlan', fn($q) => $q->where('schoolyear_id', $request->schoolyear_id));
        if ($request->filled('status'))        $query->where('status', $request->status);

        return response()->json($query->orderByDesc('account_id')->paginate($request->input('per_page', 20)));
    }

    public function store(StoreStudentFeePlanRequest $request)
    {
        try {
            $account = $this->service->create($request->validated());
            return response()->json($account->load(['student.user', 'feePlan.schoolYear']), 201);
        } catch (\RuntimeException $e) {
            return response()->json(['message' => $e->getMessage()], 422);
        }
    }

    public function show(int $id)
    {
        return response()->json(
            StudentFeePlan::where('account_id', $id)
                ->with(['student.user', 'feePlan.schoolYear', 'invoices'])
                ->firstOrFail()
        );
    }

    public function update(UpdateStudentFeePlanRequest $request, int $id)
    {
        $account = StudentFeePlan::where('account_id', $id)->with('feePlan')->firstOrFail();
        $this->service->update($account, $request->validated());
        return response()->json($account->load(['student.user', 'feePlan.schoolYear']));
    }

    public function destroy(int $id)
    {
        try {
            $this->service->delete(StudentFeePlan::where('account_id', $id)->firstOrFail());
            return response()->json(['message' => 'Student fee account removed successfully.']);
        } catch (\RuntimeException $e) {
            return response()->json(['message' => $e->getMessage()], 422);
        }
    }
}
